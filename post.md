This package implements a cell list methods via [spatial hash tables](https://matthias-research.github.io/pages/publications/tetraederCollision.pdf) in Julia. 

The main application is iterations over pairs of closeby particles to compute short-range **interaction forces between particles** for large particle systems. The aim is similar to that of the excellent [CellListMap.jl](https://github.com/m3g/CellListMap.jl) package. 

As of now, the main features of [SpatialHashTables.jl](https://github.com/SteffenPL/SpatialHashTables.jl) are:
- Supports **unbounded domains** and also bounded domains,
- GPU implementation possible (but work in progress...),
- `for`-loop style iteration,
- Only 200 lines of code. :D

Features which are currently not present:
- GPU performance not optimal yet,
- No support for periodic domains,
- Each pair is visited twice (i.e. `(i,j)` and `(j,i)`),
- User has to write parallelization code (see below).

## How does it work?

 The main idea is to hash functions $\mathbb{R}^{d} \to \{1,\dots,\texttt{tablesize}\}$ to partition an unbounded domain into `tablesize` many unions of axis aligned boxes. See [here](https://matthias-research.github.io/pages/publications/tetraederCollision.pdf) for details of the method. Spatial hash tables are well-known and popular in the computer graphics community, for example most recently as a feature of the [NVIDIA Python Warp](https://developer.nvidia.com/warp-python) framework.

On bounded domains, we use the same datastructure as proposed for spatial hash tables, which is different than the datastructure used in [CellListMap.jl](https://github.com/m3g/CellListMap.jl).

## Is it fast?

Below are a few examples for serial CPU, multithreaded CPU and GPU usage. For scripts, timings an a short comparison with CellListMap, check out the full scripts [here](https://github.com/SteffenPL/SpatialHashTables.jl/tree/main/benchmarks).

In short: 
- _On bounded domains:_ The new `BoundedHashTable` type has similar speed as `PeriodicSystem` from CellListMap.jl.
- _On unbounded domains:_ The `SpatialHashTable` is only a bit slower than the bounded counterpart, but provides more flexibility and will be preferable in sparse cases and whenever bounds are missing.

GPU performance on my system `10x slower` than CPU multithreading. I'm working on it ;) 

## Example usage

Below, we compute $2 \sum_{i,j} \Vert X_i - X_j \Vert$ for given points $X_1,\dots,X_N \in \mathbb{R}^3$. The code can easily be adapted for other computations.

### Setup

```julia
using StaticArrays, SpatialHashTables

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

If the positions change, one can simply call `updateboxes!` to update the hash table. This is done in $\mathcal{O}(n)$ time and allocation free:
```julia
updateboxes!(bht, X)  # bounded domains; error if X[i] outside!
updateboxes!(sht, X)  # unbounded domains
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

The user has to write his own parallelization code. This provides in return more flexibility and control, but requires a bit more work.
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
res = sum(res_mt)
```

### GPU

Essentially the same pattern as before applies also for GPU code. At the moment only `BoundedHashTable` is supported on the GPU.
```julia
using CUDA
using CUDA: i32 

# convert to GPU friendly types
X_gpu = cu(X)
ht_gpu = cu(bht)

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


threads = 256
blocks = cld(N, threads)

res_gpu = CUDA.zeros(length(X))
@cuda threads=threads blocks=blocks gpu_kernel!(bht_gpu, X_gpu, r, res_gpu)
res = sum(res_gpu)
```
---
PS: @lmiq If some of these things are interesting for your package, let me know. It was easier for me to test this from scratch, but in principle I would be happy if the good parts from this idea merge into CellListMap.jl.