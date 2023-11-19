This package implements a cell list method via [spatial hash tables](https://matthias-research.github.io/pages/publications/tetraederCollision.pdf) in Julia.

The primary aim is to speed up computations of short-range interactions between particles in large particle systems, e.g. for terms like $F_i = \sum_{j} F(X_i,X_j)$ for given $X_1,\dots,X_n \in \mathbb{R}^d$ and provided that $F(X_i,X_j) \approx 0$ for distances larger than a cutoff radius $\Vert X_i - X_j \Vert > R$.

The aim is similar to that of the excellent [CellListMap.jl](https://github.com/m3g/CellListMap.jl) package. (I'm also open to merge eventually, it was just easier to try the idea in a new package first.)

As of now, the main features of [SpatialHashTables.jl](https://github.com/SteffenPL/SpatialHashTables.jl) are:
- Support for **unbounded domains** and also bounded domains,
- No allocations, neither during update of positions nor at iteration of all neigbours. 
- Can be used on **GPUs** (but still a bit slow),
- Minimal but flexible interface, arbitrary dimension, fast CPU multi-threading.
- Only 200 lines of code (without docs). 😎

## Is it fast?

In short: 
- On bounded domains, the implemenation has **similar performance** as CellListMap.jl, see [benchmarks](https://github.com/SteffenPL/SpatialHashTables.jl/tree/main/benchmarks).

- For unbounded domains I lack a reference to compare to, but it seems to be reasonable fast.

- GPU performance is not optimal yet. On my system `10x slower` than the CPU. I'm working on it ;) 

## Example usage

Please check out [readme](https://github.com/SteffenPL/SpatialHashTables.jl) and the 
[docs](https://SteffenPL.github.io/SpatialHashTables.jl/dev/) for more details.

Below, we compute $F_i = \sum_{j} V(\Vert X_j - X_i \Vert) (X_j - X_i)$ for given points $X_1,\dots,X_N \in \mathbb{R}^3$ and a function $V(d) = 1/d^3$. Note that we only need to provide a `tablesize` (number of hashes) and a `cellsize` (defines binning of particles into cells). No bounds needed, which is often convenient.

```julia
using StaticArrays, SpatialHashTables

N = 1_000_000
X = randn(SVector{3, Float64}, N)
F = randn(SVector{3, Float64}, N)
V(d) = 1/d^3

r = 1/N^(1/3)

cellsize = 1/N^(1/3)
tablesize = 1_000_000
ht = SpatialHashTable(X, cellsize, tablesize)

Threads.@threads for i in eachindex(X)
    for j in neighbours(ht, X[i], r)
        d² = sum(x -> x^2, X[i] - X[j])
        if 0 < d² < r^2
            d = sqrt(d²)
            F[i] += V(d) * (X[i] - X[j])
        end
    end
end

# Timing: 0.230324 seconds (190 allocations: 20.023 KiB)
# (note that allocations stem from parallelization, serial version is allocation free)
```

I hope this package is useful for the Julia ecosystem. If you have any questions, please open an issue or contact me.
