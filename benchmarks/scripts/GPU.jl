using StaticArrays, SpatialHashTables, CellListMap, Test, BenchmarkTools
using CUDA
using CUDA: i32

include("setup.jl")  # defines: N, X, 
N, Dim, r, X = setup(100_000, 3, dtype = Float32)

# make GPU friendly
X_gpu = cu(X)

bht = BoundedHashTable(X, r, ones(Dim))
bht_gpu = cu(bht)

function gpu_kernel!(ht, X, r, res)
    index = (blockIdx().x - 1i32) * blockDim().x + threadIdx().x
    stride = gridDim().x * blockDim().x

    for i in index:stride:Int32(length(X))
        res[i] = 0.0f0
        for j in neighbours(ht, X[i], r)        
            d2 = dist_sq(X[i], X[j])  
            res[i] += ifelse(i < j && d2 < r^2, 1.0f0, 0.0f0)
        end
    end
    return nothing
end

res = CUDA.zeros(Float32, length(X))

function test_gpu(bht_gpu, X_gpu, r, res)
    kernel = @cuda launch = false  gpu_kernel!(bht_gpu, X_gpu, r, res)
    config = launch_configuration(kernel.fun)
    threads = min(length(X_gpu), config.threads)
    blocks = cld(length(X_gpu), threads)

    CUDA.@sync begin 
        kernel(bht_gpu, X_gpu, r, res; threads, blocks)
    end
    return sum(res)
end



function test_parallel(ht, X, r)
    e = zeros(Threads.nthreads())
    Threads.@threads for i in eachindex(X)
        tid = Threads.threadid()
        for j in neighbours(ht, X[i], r)
            if i < j
                d2 = dist_sq(X[i], X[j])
                if d2 < r^2
                    e[tid] += 1.0f0
                end
            end
        end
    end
    return sum(e)
end

#@time test_gpu(bht_gpu, X_gpu, r, res)

@test test_parallel(bht, X, r) ≈ test_gpu(bht_gpu, X_gpu, r, res)


@time test_parallel(bht, X, r)
# 0.004972 seconds (243 allocations: 32.391 KiB)

@time test_gpu(bht_gpu, X_gpu, r, res)
# 0.073235 seconds (258 allocations: 13.844 KiB)


# kernel = @cuda launch = false  gpu_kernel!(bht_gpu, X_gpu, r, res)
# config = launch_configuration(kernel.fun)
# threads = min(length(X_gpu), config.threads)
# blocks = cld(length(X_gpu), threads)

#CUDA.@profile trace=true @cuda threads=threads blocks=blocks gpu_kernel!(bht_gpu, X_gpu, r, res)