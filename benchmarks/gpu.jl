using Revise
using StaticArrays
using SpatialHashTables
using CellListMap
using Test
using BenchmarkTools
using ChunkSplitters
using Base.Threads

@inline dist_sq(x,y) = sum(abs2, x - y)
@inline function energy(x, y, i, j, d2, u, cutoff)
    if d2 < cutoff^2
        u += dist_sq(x,y)
    end
    return u 
end

function innerloop(grid, X, r, i, Xi, e)
    for j in HashGridQuery(grid, X[i], r)
        if i < j
            e = energy(X[i], X[j], i, j, dist_sq(X[i], X[j]), e, r)
        end
    end
    return e
end

function batch(grid, X, r, i, e)
    tid = Threads.threadid()
    Xi = X[i]
    ei = 0.0
    @inbounds for j in HashGridQuery(grid, Xi, r)
        if j < i
            Xj = X[j]
            ei = energy(Xi, Xj, i, j, dist_sq(Xi, Xj), ei, r)
        end
    end
    e[tid] += ei 
    return nothing
end

function test_parallel(grid, X, r)
    nchunks = Threads.nthreads()
    e = zeros(nchunks)
    Threads.@threads for i in eachindex(X)
        batch(grid, X, r, i, e) 
    end
    return sum(e)
end

function test_naive(X, r)
    e = 0.0
    for i in eachindex(X)
        for j in 1:i
            e = energy(X[i], X[j], i, j, dist_sq(X[i], X[j]), e, r)
        end
    end
    return e
end

function run_celllistmap(box,cl)
    u = CellListMap.map_pairwise!(
        (x,y,i,j,d2,u) -> energy(x,y,i,j,d2,u,box.cutoff), 
        0.0, box, cl
    ) 
    return u
end

x, _ = CellListMap.xatomic(10^5)
box = Box(limits(x), 12.0)
cl = CellList(x, box)

N = length(x)
r = box.cutoff
Dim = length(eltype(x))
X = x
ht = HashGrid(X, r, Tuple(box.nc))
updatecells!(ht, X)

using CUDA
Xg = cu(SVector{3,Float32}.(X))
cellwidth = Float32.(r) 
gridsize = Int32.(Tuple(box.nc))
grid = HashGrid{CuVector{Int32}}(Xg, cellwidth, gridsize, nthreads = 1024)
updatecells!(grid, Xg)

using KernelAbstractions

@kernel function inner_gpu!(grid, X, e, r)
    i = @index(Global)
    Xi = X[i]

    e0 = 0.0f0
    for j in HashGridQuery(grid, Xi, r)         
        Xj = X[j]
        if i < j
            e0 = energy(Xi, Xj, i, j, dist_sq(Xi, Xj), e0, r)
        end
    end
    e[i] = e0
end


e = CUDA.zeros(length(X))

function parallel_gpu(grid, X, r, e)
    inner_gpu!(grid.backend, 1024)(grid, X, e, r, ndrange = length(X))
    KernelAbstractions.synchronize(grid.backend)
    return e
end

# @time parallel_gpu(grid, Xg, cellwidth, e)
# @time run_celllistmap(box, cl)

parallel_gpu(grid, Xg, cellwidth, e)
@time parallel_gpu(grid, Xg, cellwidth, e)

CUDA.@profile parallel_gpu(grid, Xg, cellwidth, e)

# res = parallel_gpu(ht, X, r)
