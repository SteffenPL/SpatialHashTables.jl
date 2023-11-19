using Revise
using StaticArrays, SpatialHashTables, CellListMap, Test, BenchmarkTools

include("setup.jl")  # defines: N, X, 
N, Dim, r, X = setup(10_000, 3)

function energy(x, y, i, j, d2, u)
    if d2 < 0.02^2
        u += dist_sq(x, y)
    end
    return u 
end

system = setup_celllistmap(X, r, energy)
system.parallel = false

bht = BoundedHashTable(X, r, ones(Dim))
sht = SpatialHashTable(X, r, 5000)

function test_serial(ht, X, r)
    e = 0.0
    for i in eachindex(X)
        for j in neighbours(ht, X[i], r)
            if i < j
                d2 = dist_sq(X[i], X[j])
                e = energy(X[i], X[j], i, j, dist_sq(X[i], X[j]), e)
            end
        end
    end
    e
end

@test test_serial(bht, X, r) ≈ map_pairwise!(energy, system)
@test test_serial(sht, X, r) ≈ map_pairwise!(energy, system)

@btime test_serial($bht, $X, $r) 
# 3.060 ms (0 allocations: 0 bytes)

@btime test_serial($sht, $X, $r) 
# 9.463 ms (0 allocations: 0 bytes)

@btime map_pairwise!($energy, $system; update_lists=false) 
# 4.147 ms (0 allocations: 0 bytes)
