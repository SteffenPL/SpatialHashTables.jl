This package implements a cell list method via [spatial hash tables](https://matthias-research.github.io/pages/publications/tetraederCollision.pdf) in Julia. 

The primary aim is to speed up computations of short-range interactions between particles in large particle systems, e.g. for terms like $F_i = \sum_{j} F(X_i,X_j)$ for given $X_1,\dots,X_n \in \mathbb{R}^d$ and provided that $F(X_i,X_j) \approx 0$ for distances larger than a cutoff radius $\Vert X_i - X_j \Vert > R$.

The aim is similar to that of the excellent [CellListMap.jl](https://github.com/m3g/CellListMap.jl) package. (I'm also open to merge eventually, it was just easier to try the idea in a new package first.)

As of now, the main features of [SpatialHashTables.jl](https://github.com/SteffenPL/SpatialHashTables.jl) are:
- Support for **unbounded domains** and also bounded domains,
- No allocations during neither during update of positions nor at iteration of all neigbours. 
- Can be used on **GPUs** (but still a bit slow),
- minimal interface, arbitrary dimension, CPU multi-threading and
- only 200 lines of code. 😎

## How does it work?

 The main idea is to use a hash function $\mathbb{R}^{d} \to \{1,\dots,\texttt{tablesize}\}$ to partition an unbounded domain into `tablesize` many unions of axis aligned boxes. See [Matthias Müller's paper](https://matthias-research.github.io/pages/publications/tetraederCollision.pdf) for details of the method.

The table of cells belonging to one hash index is in turn stored as a indirect list, see for example [Carmen Cincotti's tutorial](https://carmencincotti.com/2022-10-31/spatial-hash-maps-part-one/). In that sense, also the implementation for the bounded case differs from [CellListMap.jl](https://github.com/m3g/CellListMap.jl).

## Is it fast?

Below are a few examples for serial CPU, multithreaded CPU and GPU usage.  [Here are the full scripts and some timings](https://github.com/SteffenPL/SpatialHashTables.jl/tree/main/benchmarks).

In short: 
- On bounded domains, the implemenation has **similar performance** as CellListMap.jl.
- Unbounded domains are naturally slower for dense particle distributions (in benchmark `3x slower`), but it becomes `faster particles are sparse`, e.g. if `66%` of the space is empty.
- Right now, GPU performance on my system `10x slower` than CPU multithreading. I'm working on it ;) 

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

## What's next?

Any feedback or suggestions are welcome. (Espcially if you know how to make the GPU code faster.)

I made the package since I want to do agent-based modelling on GPUs, so, the GPU performance and eventually automatic differentiation are my next goals.

And, like always, I need to work on the documentation. 😅