using CUDA

X

X_gpu = cu(X)

function gpu_atan_pairs(red, st, X, cutoff)
    index = threadIdx().x 
    stride = blockDim().x

    for i in index:stride:length(X)
        for j in neighbours(st, X[i], cutoff)
            if i < j
                d2 = sum( x -> x^2, X[i] - X[j])
                if d2 < cutoff^2
                    red[index] += atan(d2)
                end
            end
        end
    end
end

@cuda threads=256 test_speed(st, X_gpu, system.cutoff)


function test_speed(st, X, cutoff)
    red = zeros(Threads.nthreads())

    Threads.@threads for i in eachindex(X)
        for j in neighbours(st, X[i], cutoff)
            if i < j
                d2 = sum( x -> x^2, X[i] - X[j])
                if d2 < cutoff^2
                    red[Threads.threadid()] += atan(d2)
                end
            end
        end
    end
    sum(red)
end
