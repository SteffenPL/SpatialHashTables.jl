# SpatialHashTables

<!--
[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://SteffenPL.github.io/SpatialHashTables.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://SteffenPL.github.io/SpatialHashTables.jl/dev/)
-->
[![Build Status](https://github.com/SteffenPL/SpatialHashTables.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/SteffenPL/SpatialHashTables.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/SteffenPL/SpatialHashTables.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/SteffenPL/SpatialHashTables.jl)

> This package is work in progress!

Creates neigbour lists (currently only for bounded domains) and allows iteration over all 
potential neigbours. 

In the future, I would like to support spatial hash tables, which allow fast neighbour detection 
without the need to specify a particular domain. 

A typical application looks like 
```julia
using SpatialHashTables 
using StaticArrays 

X = rand(SVector{2, Float64}, 100)
domain = (min = SVector{2, Float64}(0, 0), max = SVector{2, Float64}(1, 1))
grid = (5, 5)

ht = SpatialHashTable(domain, grid, X)

# the structure can also be resized and updated 
X = rand(SVector{2, Float64}, 1000)
resize!(ht, length(X))
updateboxes!(ht, X)

# computing interaction terms
R = 0.1  # interaction radius
F = @SVector [0.0, 0.0]
for i in eachindex(X) 
    for j in neighbours(ht, X[i], R)
        if i < j
            F += ( X[i] - X[j] )  # replace this with the computation your are interested in
        end
    end
end
```

The operations for updating lists and finding neighbours are allocation free.
```julia 
using BenchmarkTools

interations(ht, X, R) = ( (i,j) for i in eachindex(X) for j in neighbours(ht, X[i], R) )
function test_allocations(ht, X, R)
    F = @SVector [0.0, 0.0]
    for (i,j) in interations(ht, X, R)
        if i < j
            F += X[i] - X[j]
        end
    end
    return F
end

@btime test_allocations($ht, $X, 0.1)   # 383.077 μs (0 allocations: 0 bytes)
```

# Similar packages

A more established and tested package for computing of interaction terms is [CellListMap.jl](https://github.com/m3g/CellListMap.jl).


# Benchmarks [work in progress]

This package is probably not faster than [CellListMap.jl](https://github.com/m3g/CellListMap.jl) 
but provides an easier interface at the moment.

```julia

