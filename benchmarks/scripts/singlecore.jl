using StaticArrays, SpatialHashTables, CellListMap

include("setup.jl")  # defines: N, X, 

N, Dim, r, X, energy = setup(1_000, 3)

system = setup_celllistmap(X, r, energy)
system.parallel = false

ht = BoundedHashTable(X, r, ones(Dim))
# ht = SpatialHashTable(X, 1000, r)

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



@assert test_serial(ht, X, r) ≈ map_pairwise!(energy, system)

@profview test_serial(ht, X, r)

using BenchmarkTools
BenchmarkTools.DEFAULT_PARAMETERS.seconds = 0.50
@btime test_serial($ht, $X, $r)
@btime map_pairwise!($energy, $system; update_lists=false)
