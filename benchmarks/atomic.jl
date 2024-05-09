using Revise
using StaticArrays
using SpatialHashTables
using CellListMap
using Test
using BenchmarkTools
using ChunkSplitters
using Base.Threads

benchmark_results = Any[]
function report(case, pkg, N, secs, backend, nthreads)
    push!(benchmark_results, (;case,pkg,N,secs,backend,nthreads))
end





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

# using CUDA
# Xg = cu(SVector{3,Float32}.(X))
# cellwidth = Float32.(r) 
# gridsize = Int32.(Tuple(box.nc))
# grid = HashGrid{CuVector{Int32}}(Xg, cellwidth, gridsize, nthreads = 1024)
# updatecells!(grid, Xg)

# using KernelAbstractions

# @kernel function inner_gpu!(grid, X, e, r)
#     i = @index(Global)
#     Xi = X[i]

#     e0 = 0.0f0
#     for j in HashGridQuery(grid, Xi, r)         
#         Xj = X[j]
#         if i < j
#             e0 = energy(Xi, Xj, i, j, dist_sq(Xi, Xj), e0, r)
#         end
#     end
#     e[i] = e0
# end


# e = CUDA.zeros(length(X))

# function parallel_gpu(grid, X, r, e)
#     inner_gpu!(grid.backend, 1024)(grid, X, e, r, ndrange = length(X))
#     KernelAbstractions.synchronize(grid.backend)
#     return e
# end

# @time parallel_gpu(grid, Xg, cellwidth, e)
# @time run_celllistmap(box, cl)

# CUDA.@profile parallel_gpu(grid, Xg, cellwidth, e)

# res = parallel_gpu(ht, X, r)

test_parallel(ht, X, r)
map_pairwise!((x,y,i,j,d2,u) -> energy(x,y,i,j,d2,u,box.cutoff), 0.0, box, cl)

@test test_naive(X, r) ≈ map_pairwise!((x,y,i,j,d2,u) -> energy(x,y,i,j,d2,u,box.cutoff), 0.0, box, cl)
@test test_parallel(ht, X, r) ≈ map_pairwise!((x,y,i,j,d2,u) -> energy(x,y,i,j,d2,u,box.cutoff), 0.0, box, cl)

@btime test_parallel($ht, $X, $r) 
@btime run_celllistmap($box, $cl)





using JET
@report_call test_parallel(ht, X, r) 





dist_sq(x,y) = sum(abs2, x - y)
function energy(x, y, i, j, d2, u, cutoff)
    if d2 < cutoff^2
        u += dist_sq(x,y)
    end
    return u 
end


function batchs(grid, X, r, i, e)
    Xi = X[i]
    ei = e
    @inbounds for j in HashGridQuery(grid, Xi, r)
        if j < i
            Xj = X[j]
            ei = energy(Xi, Xj, i, j, dist_sq(Xi, Xj), ei, r)
        end
    end
    return ei 
end


@check_allocs function test_serial(grid, X, r)
    e = zeros(1)
    for i in eachindex(X)
        e = batchs(grid, X, r, i, e) 
    end
    return sum(e)
end

using AllocCheck

test_serial(ht, X, r)

@time test_serial(ht, X, r)

@profview test_serial(ht, X, r)

using JET 
@report_call test_serial(ht, X, r)


1












function batch(grid, X, r, i, Xi, ei)
    for j in HashGridQuery(grid, Xi, r)
        Xj = X[j]
        ei = energy(Xi, Xj, i, j, dist_sq(Xi, Xj), ei, r)
    end
    return ei
end

@inbounds function test_cpu_parallel(grid, X, r)
    nchunks = Threads.nthreads()
    e = zeros(nchunks)
    Threads.@threads :static for i in eachindex(X) 
        ichunk = Threads.threadid()
        ei = 0.0
        Xi = X[i]
        ei = batch(grid, X, r, i, Xi, ei)
        e[ichunk] += ei
    end
    return sum(e)
end

@inbounds function test_cpu_serial(grid, X, r)
    u = 0.0
    for i in eachindex(X) 
        Xi = X[i]
        u = batch(grid, X, r, i, Xi, u)
    end
    return u
end

@inbounds function test_cpu_naive(grid, X, r)
    u = 0.0
    for i in eachindex(X) 
        Xi = X[i]
        for j in eachindex(X)
            if i < j
                u = energy(Xi,X[j],i,j,dist_sq(Xi,X[j]),u,r)
            end
        end
    end
    return u
end

function test_clm(box, cl)
    u = map_pairwise!( 
        (x,y,i,j,d2,u) -> energy(x,y,i,j,d2,u,box.cutoff),
        0.0, box, cl
    )
    return u
end

SVec3d = SVector{3,Float64}
SVec3f = SVector{3,Float32}

x_cpu, box = CellListMap.xatomic(10^4)

r = box.cutoff
cellwidth = box.cell_size
gridsize = Tuple(box.nc)

system = ParticleSystem(xpositions = x_cpu, 
                    unitcell = cellwidth .* 3,
                    cutoff = 12.0,
                    output = 0.0)

@time grid = HashGrid(x_cpu, cellwidth, gridsize; origin = -cellwidth)
@time updatecells!(grid, x_cpu)


@time test_clm(box, cl)
@time test_cpu_parallel(grid, x_cpu, r)
@time test_cpu_serial(grid, x_cpu, r)
@time test_cpu_naive(grid, x_cpu, r)

@profview test_cpu_serial(grid, x_cpu, r)
@profview test_clm(box, cl)


map_pairwise!((x,y,i,j,d2,u) -> energy(x,y,i,j,d2,u,box.cutoff), 0.0, box, cl)

using JET
@report_call test_cpu_parallel(grid, x_cpu, r)



using CellListMap
using Test


let 
    X, box = CellListMap.xatomic(10^4)
    cl = CellList(X, box)
    cutoff = box.cutoff / 2

    dist_sq(x,y) = sum(abs2, x - y)
    function energy(x, y, i, j, d2, u)
        if d2 < cutoff^2
            u += dist_sq(x,y)
        end
        return u 
    end

    function naive(X)
        u = 0.0
        for i in eachindex(X) 
            Xi = X[i]
            for j in eachindex(X)
                if i <= j
                    u = energy(Xi,X[j],i,j,dist_sq(Xi,X[j]),u)
                end
            end
        end
        return u
    end

    @test naive(X) ≈ map_pairwise(energy, 0.0, box, cl)
end