```@meta
CurrentModule = SpatialHashTables
```

# SpatialHashTables

Documentation for [SpatialHashTables](https://github.com/SteffenPL/SpatialHashTables.jl).

## Core interface

There are essentially four functions which are needed to use the package:


The typical constructors have inputs which determine: 
- the number of particles or the particles themselves
- the spatial size of the cells
- the number of hash indices (only for `SpatialHashTable`)
- the domain (only for `BoundedHashTable`)

The core functions are `BoundedHashTable` and `SpatialHashTable` which create the hash tables.
The hash tables are then updated via `updatetable!` and once can iterate
neigbours of a given position with the `neighbours` function.

### BoundedHashTable

The easiest constructor is `BoundedHashTable(X, r, domain)`, where `X` is a vector of particle positions, `r` is the spatial size of the cells, and `domain` is the upper bounde for the domain of the particles.
```julia
using StaticArrays, SpatialHashTables
bht = BoundedHashTable(100, 0.1, (1.0, 2.0))
``` 

To update the table for given positions, one can call `updatetable!`:
```julia
X = rand(SVector{Float64,3}, 100)
updatetable!(bht, X)
```
where `X` is a vector of particle positions. This is done in $\mathcal{O}(n)$ time and allocation free.

Iteration over all neigbours of a given position `x` is done via `neighbours(bht, p, r)`:
```julia
p = @SVector [0.5, 0.5, 0.5]
r = 0.2

for j in neighbours(bht, p, r)
    d² = sum(x -> x^2, X[j] - p)

    if d² < r^2
        # do something with X[j]
    end
end
```
where `p` is a position and `r` is the cutoff distance. Notice that the cutoff distance 
can be larger than the cell size during construction. In this case more cells will be visited
during the iteration.

### SpatialHashTable

The constructor `SpatialHashTable(X, tablesize, r)` is similar to `BoundedHashTable`, but does not require a domain
but instead the number of hash indices `tablesize`:
```julia
using StaticArrays, SpatialHashTables

X = rand(SVector{Float64,3}, 1000)
tablesize = 1000
sht = SpatialHashTable(X, 0.1, tablesize)
```
This creates a hash table with `tablesize` many hash indices. The number of hash indices should be chosen such that the number of particles per hash index is not too small. Due to the hashing, it can happen that 
particles with large distance are assigned to the same hash index. This can be avoided by choosing a larger `tablesize`.

Too large `tablesize` can lead to a large memory footprint.

Moreover, by providing the positions `X` instead of the number of particles as the first input,
the constructor will automatically call `updatetable!` to update the hash table. 
The same holds for constructors of `BoundedHashTable`.

Using the hash table is similar to `BoundedHashTable`. Let us iterate here over all 
pairs of points within a given radius `r`:
```julia
r = 0.05
for i in eachindex(X)
    for j in neighbours(sht, X[i], r)
        d² = sum(x -> x^2, X[i] - X[j])

        if d² < r^2
            # do something with X[j]
        end
    end
end
```

## CPU Parallelization

The above code snippets are for serial CPU computations. 
For parallel computations, one has to write small parallelization boilder code. 
However, this approach provides in return flexibility and control.

Provided that `julia` is started with multiple threads, one can use `Threads.@threads` to parallelize the code:
```julia
res_mt = zeros(Threads.nthreads())

Threads.@threads for i in eachindex(X)
    for j in neighbours(sht, X[i], r)
        d² = sum(x -> x^2, X[i] - X[j])

        if d² < r^2
            res_mt[Threads.threadid()] += sqrt(d²)
        end
    end
end

res = sum(res_mt)
```
where `res_mt` is a vector of length `Threads.nthreads()` which stores the partial results of each thread.
The final result is then the sum of the partial results.

This simple code snippet can be easily adapted to other computations. One example would be 
to compute forces for all particles given a potential function `V`:
```julia
F = zeros(SVector{Float64,3}, length(X))
V(d) = 1/d^3

Threads.@threads for i in eachindex(X)
    for j in neighbours(sht, X[i], r)
        d² = sum(x -> x^2, X[i] - X[j])

        if d² < r^2
            F[i] += (X[i] - X[j]) * V(sqrt(d²))
        end
    end
end
```

Note that the code runs over all particles `i` and then over all neighbours `j` of `i`.
For some computations one might add a check if `i == j` to avoid self-interactions
or `i < j` to avoid double counting.

## GPU Parallelization

The package is not fully optimized for GPUs. 
However, here is a simple example how to use the package on GPUs via `CUDA.jl`:

```julia
using CUDA, StaticArrays, SpatialHashTables

X = rand(CuArray{SVector{Float64,3}}, 1000)
cellsize = 1/N^(1/3)

ht = BoundedHashTable(X, cellsize, (1.0, 1.0, 1.0))

r = 0.05

X_gpu = cu(X)
bht_gpu = cu(ht)

function gpu_kernel!(ht_gpu, X, r, res)
    index = (blockIdx().x - 1i32) * blockDim().x + threadIdx().x
    stride = gridDim().x * blockDim().x

    for i in index:stride:Int32(length(X))
        res[i] = 0.0f0
        for j in neighbours(ht_gpu, X[i], r)        
            d2 = dist_sq(X[i], X[j])  
            res[i] += ifelse(i < j && d2 < r^2, 1.0f0, 0.0f0)
        end
    end
    return nothing
end

res = CUDA.zeros(Float32, length(X))

function compute_on_gpu(ht_gpu, X_gpu, r, res)

    # determine number of threads and blocks
    kernel = @cuda launch = false  gpu_kernel!(ht_gpu, X_gpu, r, res)
    config = launch_configuration(kernel.fun)
    threads = min(length(X_gpu), config.threads)
    blocks = cld(length(X_gpu), threads)

    # run kernel
    CUDA.@sync begin 
        kernel(ht_gpu, X_gpu, r, res; threads, blocks)
    end

    # reduce step
    return sum(res)
end
```

At this point in thime, the performance of the GPU code is not better than of parallel CPU code.

