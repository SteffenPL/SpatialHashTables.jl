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

A typical application looks like 
```julia
using SpatialHashTables 
using StaticArrays 

X = rand(SVector{2, Float64}, 100)
domain = (min = SVector{2, Float64}(0, 0), max = SVector{2, Float64}(1, 1))
grid = (5, 5)

ht = SpatialHashTable(domain, grid, length(X))
updateboxes!(ht, X)

X = rand(SVector{2, Float64}, 1000)

resize!(ht, length(X))
updateboxes!(ht, X)

R = 0.1
closeby_pairs = ( (i,j) for i in eachindex(X) for j in neighbours(ht, X[i], R) if i < j )

F = @SVector [0.0, 0.0]
for (i,j) in closeby_pairs 
    F += ( X[i] - X[j] )  # replace this with the computation your are interested in
end
```

# Similar packages

A more established and tested package for computing of interaction terms is [CellListMap.jl](https://github.com/m3g/CellListMap.jl).