using StaticArrays, SpatialHashTables, CellListMap, Test

include("setup.jl")  # defines: N, X, 

N, Dim, r, X = setup(10_000, 3)

function energy(x, y, i, j, d2, u) 
    if 0 < d2 < 0.02^2
        u += dist_sq(x, y)
    end
    return u 
end

system = setup_celllistmap(X, r, energy)
system.parallel = false

ht = BoundedHashTable(X, r, ones(Dim))
ht = SpatialHashTable(X, 500, r)

function test_serial(ht, X, r)
    e = 0.0
    for i in eachindex(X)
        for j in neighbours(ht, X[i], r)
            if i < j
                d2 = dist_sq(X[i], X[j])
                e = energy(X[i], X[j], i, j, d2, e)
            end
        end
    end
    e
end
@test test_serial(ht, X, r) ≈ map_pairwise!(energy, system)

collect(neighbours(ht, X[1], r))

using BenchmarkTools
@btime test_serial($ht, $X, $r)
@btime map_pairwise!($energy, $system; update_lists=false)
