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


# test performance 
function compute!(X, F, ht, r)
    F .*= 0
    Threads.@threads for i in eachindex(X)
        for j in neighbours(ht, X[i], r)
            d² = sum(x -> x^2, X[i] - X[j])
            if 0 < d² < r^2
                d = sqrt(d²)
                F[i] += V(d) * (X[i] - X[j])
            end
        end
    end
    return F 
end

compute!(X, F, ht, r)            # warmup
@time compute!(X, F, ht, r)
#  0.234256 seconds (192 allocations: 20.055 KiB)

# N vs time
# function setup(N)
#     X = randn(SVector{3, Float64}, N)
#     F = randn(SVector{3, Float64}, N)
#     V(d) = 1/d^3

#     r = 1/N^(1/3)

#     cellsize = 1/N^(1/3)
#     tablesize = 1_000_000
#     ht = SpatialHashTable(X, cellsize, tablesize)
#     return X, F, ht, r
# end

# function test_timing(N)
#     X, F, ht, r = setup(N)
#     compute!(X, F, ht, r)
#     @elapsed compute!(X, F, ht, r)
# end

# Ns = 10 .^ (3:7)
# times = test_timing.(Ns)
# 5-element Vector{Float64}:
#   0.000146
#   0.0011583
#   0.0082612
#   0.231798
#  11.4590648
#
#  -> not sure if the benchmark is correct... maybe the computation is not fair for the different Ns
