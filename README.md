# SpatialHashTables
<!-- 
[![Docs](https://img.shields.io/badge/docs-stable-blue.svg)](https://SteffenPL.github.io/SpatialHashTables.jl/stable/) -->

[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://SteffenPL.github.io/SpatialHashTables.jl/dev/)

[![Build Status](https://github.com/SteffenPL/SpatialHashTables.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/SteffenPL/SpatialHashTables.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/SteffenPL/SpatialHashTables.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/SteffenPL/SpatialHashTables.jl)

## Aim 

Creates spatial hash tables on bounded and unbounded domains for fast 
computations of pairwise interaction terms. Updating the hash table is 
done in $\mathcal{O}(n)$ time and allocation free. Iteration over neigbour lists 
is also allocation free and supports CPU and GPU parallelization.

The interface is minimalistic. For a more full-featured and established package, see 
[CellListMap.jl](https://github.com/m3g/CellListMap.jl).


## Example 

```julia
using SpatialHashTables, StaticArrays
const SVec3 = SVector{3, Float64}
dist_sq(a,b) = sum(x->x^2, a-b)

N = 10_000
X = rand(SVec3, N)
r = 1 / N^(1/3)	
hash_table = SpatialHashTable(X, 5000, r)

function test_serial(ht, X, r)
    e = 0.0
    for i in eachindex(X)
        for j in neighbours(ht, X[i], r)
            d2 = dist_sq(X[i], X[j])
            if d2 < r^2
                e += sqrt(d2)  
            end
        end
    end
    return e
end

function test_parallel(ht, X, r)
    e = zeros(Threads.nthreads())
    Threads.@threads for i in eachindex(X)
        tid = Threads.threadid()
        for j in neighbours(ht, X[i], r)
            d2 = dist_sq(X[i], X[j])
            if d2 < r^2
                e += sqrt(d2)  
            end
        end
    end
    return sum(e)
end
```