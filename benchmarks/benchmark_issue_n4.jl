using Revise
using StaticArrays
using SpatialHashTables
using CellListMap
using Test
using BenchmarkTools
using ChunkSplitters
using Base.Threads
using LinearAlgebra

dist_sq(x,y) = sum(abs2, x - y)

function energy(x, y, i, j, d2, u, cutoff)
    if d2 < cutoff^2
        u += 1.0
        u += 1e-9 * dist_sq(x,y)
    end
    return u 
end

function test_parallel(ht, X, r)
    nchunks = Threads.nthreads()
    e = zeros(nchunks)
    @sync for (irange, ichunk) in chunks(X, nchunks)
        @spawn for i in irange 
            for j in neighbours(ht, X[i], r)
                if i < j
                    e[ichunk] = energy(X[i], X[j], i, j, dist_sq(X[i], X[j]), e[ichunk], r)
                end
            end
        end
    end
    return sum(e)
end

x, _ = CellListMap.xatomic(10^5)
box = Box(limits(x), 12.0)
cl = CellList(x, box)

N = length(x)
r = box.cutoff
Dim = length(eltype(x))
X = x
bht = BoundedHashTable(X, r, limits(x).limits .+ 12 ) # this errors
sht = SpatialHashTable(X, r, 7^3)

@test test_parallel(bht, X, r) ≈ map_pairwise!((x,y,i,j,d2,u) -> energy(x,y,i,j,d2,u,box.cutoff), 0.0, box, cl)
@test test_parallel(sht, X, r) ≈ map_pairwise!((x,y,i,j,d2,u) -> energy(x,y,i,j,d2,u,box.cutoff), 0.0, box, cl)

@btime test_parallel($bht, $X, $r) 

@btime test_parallel($sht, $X, $r) 

function run_celllistmap(box,cl)
    u = CellListMap.map_pairwise!(
        (x,y,i,j,d2,u) -> energy(x,y,i,j,d2,u,box.cutoff), 
        0.0, box, cl
    ) 
    return u
end
@btime run_celllistmap($box, $cl)


function test_serial(ht, X, r)
    e = 0.0
    for i in eachindex(X) 
        for j in neighbours(ht, X[i], r)
            if i < j
                e = energy(X[i], X[j], i, j, dist_sq(X[i], X[j]), e, r)
            end
        end
    end
    return e
end

run_celllistmap(box, cl) 
@btime test_serial($bht, $X, $r) 
test_parallel(bht, X, r)

@profview test_serial(bht, X, r) 

using SpatialHashTables: gridindices, hashindex

function test_parallel_(ht, X, r)
    nchunks = Threads.nthreads()
    e = zeros(nchunks)

    Dim = 3
    widths = @. ceil(Int, r * ht.inv_cellsize)
    neighbour_indices = CartesianIndices(ntuple(i -> -1:1, Dim))

    #@sync for (irange, ichunk) in chunks(X, nchunks)
    #    @spawn for i in irange 
        ichunk = 1
    @inbounds for i in eachindex(X)
            Xi = X[i]
            gridpos = gridindices(ht, Xi)
            
            for boxoffset in neighbour_indices

                boxrep = gridpos .+ Tuple(boxoffset)
                if all(@. 1 <= boxrep <= ht.gridsize)
                    boxhash = hashindex(ht, boxrep)
                    box_start = ht.cellcount[boxhash] + 1 
                    box_end = ht.cellcount[boxhash+1]
                    
                    for k in box_start:box_end
                        j = ht.particlemap[k]
                        Xj = X[j]
                        if i < j 
                            e[ichunk] = energy(Xi, Xj, i, j, dist_sq(Xi, Xj), e[ichunk], r)
                        end
                    end
                end
            end
        #end
    end
    return sum(e)
end

@time test_parallel_(bht, X, r) 
@profview test_parallel_(bht, X, r)
# 0.824237 seconds (253 allocations: 35.656 KiB)