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

The interface is minimalistic. For a more full-featured and established package, see [CellListMap.jl](https://github.com/m3g/CellListMap.jl).

See the [benchmarks](https://github.com/SteffenPL/SpatialHashTables.jl/tree/main/benchmarks) folder for some timings and performance comparisons.

## How does it work?

*The package is currently undergoing many changes. Documentation will be updated soon.*

## Documentation 

Please let me know if the [documenation](https://SteffenPL.github.io/SpatialHashTables.jl/dev/) is unclear or if you have any suggestions for improvements.
