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
bht = BoundedHashTable(X, r, ones(Dim))
sht = SpatialHashTable(X, r, 5000)

function test_parallel(ht, X, r)
    e = zeros(Threads.nthreads())

    Threads.@threads for i in eachindex(X)
        tid = Threads.threadid()
        for j in neighbours(ht, X[i], r)
            if i < j
                e[tid] = energy(X[i], X[j], i, j, dist_sq(X[i], X[j]), e[tid])
            end
        end
    end
    return sum(e)
end

@test test_parallel(bht, X, r) ≈ map_pairwise!(energy, system)
@test test_parallel(sht, X, r) ≈ map_pairwise!(energy, system)

@btime test_parallel($bht, $X, $r) 
# 497.320 μs (242 allocations: 32.38 KiB)

@btime test_parallel($sht, $X, $r) 
# 1.431 ms (242 allocations: 29.38 KiB)

@btime map_pairwise!($energy, $system; update_lists=false) 
# 255.188 μs (266 allocations: 50.25 KiB)
