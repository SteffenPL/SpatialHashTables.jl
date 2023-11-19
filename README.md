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


## Example usage

Below, we compute $\mathtt{res} = 2 \sum_{i,j} \Vert X_i - X_j \Vert$ for given points $X_1,\dots,X_N \in \mathbb{R}^3$. The code can easily be adapted for other computations.

### Setup

```julia
using StaticArrays, SpatialHashTables
dist_sq(a,b) = sum(x -> x^2, a-b) 

N = 10_000
r = 1/N^(1/3)
X = rand(SVector{Float64,3}, N)

domain = (1.0, 1.0, 1.0)  # only needed for bounded domains
bht = BoundedHashTable(X, r, domain)  
#         r : spatial size of cells, e.g. [0,r] x [0,r] x [0,r]

tablesize = 5000
sht = SpatialHashTable(X, tablesize, r)
# tablesize : the number of partitions/hash indices
#         r : spatial size of cells
```
There are also more general constructors. See the [docs](https://SteffenPL.github.io/SpatialHashTables.jl/dev/).

### Updating the hash table

If the positions change, one can simply call `updatetable!` to update the hash table. This is done in $\mathcal{O}(n)$ time and allocation free:
```julia
updatetable!(bht, X)  # bounded domains; error if X[i] outside!
updatetable!(sht, X)  # unbounded domains
```

### CPU Serial

```julia
res = 0.0
for i in eachindex(X)
    for j in neighbours(sht, X[i], r)
        d = sum(x -> x^2, X[i] - X[j])
        if d < r^2
           res += sqrt(d)  # ... main computation
        end
    end
end
```

### CPU Parallel

User have to write their own parallelization code. This provides in return more flexibility and control, but requires a bit more work.
```julia
res_mt = zeros(Threads.nthreads())
Threads.@threads for i in eachindex(X)
    for j in neighbours(sht, X[i], r)
        d = sum(x -> x^2, X[i] - X[j])
        if d < r^2
           res_mt[i] += sqrt(d)  # ... main computation
        end
    end
end
res = sum(res_mt)  # reduce step between threads
```

### GPU

Essentially the same pattern as before applies also for GPU code. Currently, the package itself is generic Julia code with only an `Adapt.adapt_structure` added for GPUs.
```julia
using CUDA
using CUDA: i32 

# convert to GPU friendly types
X_gpu = cu(X)
ht_gpu = cu(bht)  # only BoundedHashTable is supported on GPU, uses Int32/Float32

function gpu_kernel!(ht, X, r, res)
    index = (blockIdx().x - 1i32) * blockDim().x + threadIdx().x
    stride = gridDim().x * blockDim().x

    for i in index:stride:Int32(length(X))
        res[i] = 0.0f0
        for j in neighbours(ht, X[i], r)        
            d2 = dist_sq(X[i], X[j]) 
            # ... main computation
            res[i] += ifelse(d2 < r^2, sqrt(d2), 0.0f0)  
        end
    end
    return nothing
end

# run on GPU:
threads = 256
blocks = cld(N, threads)

res_gpu = CUDA.zeros(length(X))
@cuda threads=threads blocks=blocks gpu_kernel!(bht_gpu, X_gpu, r, res_gpu)
res = sum(res_gpu)
```