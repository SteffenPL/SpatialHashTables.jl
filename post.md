This package implements spatial hash tables in Julia. 

The main application is to compute fast short-range interaction forces between particles while avoiding $\mathcal{O}(N^2)$ runtime.

Obviously, the excellent CellListMap.jl already exists, and it solves this problem with only a slightly different approach. Therefore, I will directly compare my package with this existing one.

As of now, the highlights of SpatialHashTables.jl are:
- Unbounded domains
- GPU support
- `for`-loop style iteration over closeby pairs of points
- Short source code (150 LOC)

Features which are currently not present:
- No support for periodic domains
- Symmetry between

Here is a CPU parallel example
```julia
using StaticArrays, SpatialHashTables

N = 10_000
r = 1/sqrt(N)
X = rand(SVector{Float64,3}, N)

tablesize = 100
spacing = @SVector[ 2*r, 2*r, 2*r ]
ht = SpatialHashTable(X, tablesize, spacing)

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

# GPU
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