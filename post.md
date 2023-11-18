This package implements [spatial hash tables](https://matthias-research.github.io/pages/publications/tetraederCollision.pdf) in Julia. 

The main application is to compute fast short-range interaction forces (like gravity, repulsion, etc...) between particles while avoiding $\mathcal{O}(N^2)$ runtime. The scope is comparable to the excellent [CellListMap.jl](https://github.com/m3g/CellListMap.jl).

Since a new package is only worth writing if it offers something new: The main difference is that SpatialHashTables.jl is designed with a more minimalistic interface and adapted to GPUs. 

As of now, the highlights of SpatialHashTables.jl are:
- Unbounded domains 
- GPU support
- `for`-loop style iteration over closeby pairs of points
- Only 150 lines of code :) 

Features which are currently not present:
- No support for periodic domains
- In parallel context: Sometimes, the user might need to implement a reduce operation manually.
- Each pair is visited twice (i.e. `(i,j)` and `(j,i)`)

## How does it work?

On the technical side, one first discretizes the spatial domain into a grid of cells $\mathbb{R}^d \to \mathbb{Z}^d$ and then uses a hash function $\mathbb{Z}^d \to \{1,\dots,q\}$ to map each point to a cell in a grid. If two cells have the same hash, it just means that one iterates over more particles than necessary, but in return, one does not need to restrict the domain size. 

This idea is well-known and popular in the computer graphics community and used for example in the examples of [NVIDIA Warp](https://developer.nvidia.com/warp-python).

## Is it fast?

Below are a few examples for single threaded CPU, multithreaded CPU and GPU usage, as well as a comparison to CellListMap.jl. See also: LINK!

## Examples 

### Setup

```julia
using StaticArrays, SpatialHashTables

N = 10_000
r = 1/sqrt(N)
X = rand(SVector{Float64,3}, N)

tablesize = 100
spacing = @SVector[ 2*r, 2*r, 2*r ]
ht = SpatialHashTable(X, tablesize, spacing)
```

### CPU Serial

```julia
# Single code
res = 0.0
for i in eachindex(X)
    for j in neighbours(ht, X[i], r)
        d = sum(x -> x^2, X[i] - X[j])
        if d < r^2
           res += 1/d
        end
    end
end
```

### CPU Parallel

```julia
# Multithreaded
res_mt = zeros(Threads.nthreads())
Threads.@threads for i in eachindex(X)
    for j in neighbours(ht, X[i], r)
        d = sum(x -> x^2, X[i] - X[j])
        if d < r^2
           res_mt[i] += 1/d
        end
    end
end
res = sum(res_mt)
```

### GPU

```julia
using CUDA
Y = cu(X)
ht_gpu = cu(ht)

function kernel(ht, X, r)
    i = (blockIdx().x-1) * blockDim().x + threadIdx().x
    if i > length(X)
        return
    end

    for j in neighbours(ht, X[i], r)
        d = sum(x -> x^2, X[i] - X[j])
        if d < r^2
           atomicAdd!(res, 1/d)
        end
    end
end

res = 0.0
@cuda threads=256 kernel(ht_gpu, Y, r)
```
---
PS: @lmiq If some of these things are interesting for your package, let me know. It was easier for me to test this from scratch, but in principle I would be happy if the good parts from this idea merge into CellListMap.jl.