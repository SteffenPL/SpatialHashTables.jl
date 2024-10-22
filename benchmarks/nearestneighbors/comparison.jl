# source code copied from kristoffer.carlsson
# https://discourse.julialang.org/t/ann-spatialhashtables-jl/106553/5

using StaticArrays

const N = 1_000_000
const X = randn(SVector{3, Float64}, N)
const V(d) = 1/d^3
const r = 1/N^(1/3)

function compute_force(Xi, Xj, V, r²)
    Xdiff = Xi - Xj
    d² = sum(x -> x^2, Xdiff)
    if 0 < d² < r²
        d = sqrt(d²)
        return V(d) * Xdiff
    end
    return zero(Xi)
end

function inner_sht(F, X, grid, i, r, r²)
    Xi = X[i]
    Fi = zero(Xi)
    for j in HashGridQuery(grid, Xi, r)
       F_ij = compute_force(Xi, X[j], V, r²)
       Fi += F_ij
    end
    F[i] = Fi
end

using SpatialHashTables
function compute_forces_sht(X, r)
    F = zeros(SVector{3, Float64}, N)
    cellsize = 1/N^(1/3)
    grid = HashGrid(X, @SVector[0.0,0.0,0.0], @SVector[1.0,1.0,1.0], cellsize; nthreads = Val(1))

    r² = r^2
    for i in eachindex(X)
        inner_sht(F, X, grid, i, r, r²)
    end
    return F
end

using KernelAbstractions
@kernel function inner_sht_parallel!(F, X, grid, r, r²)
    i = @index(Global)
    Xi = X[i]
    Fi = zero(Xi)
    for j in HashGridQuery(grid, Xi, r)
       F_ij = compute_force(Xi, X[j], V, r²)
       Fi += F_ij
    end
    F[i] = Fi
end


function compute_forces_sht_parallel(X, r)
    F = zeros(SVector{3, Float64}, N)
    cellsize = 1/N^(1/3)
    grid = HashGrid(X, @SVector[0.0,0.0,0.0], @SVector[1.0,1.0,1.0], cellsize; nthreads = Val(24))

    r² = r^2
    inner_sht_parallel!(grid.backend)(F, X, grid, r, r², ndrange = length(F))
    synchronize(grid.backend)
    return F
end



function inner_kdtree(F, X, kdtree, i, r, r²)
    Xi = X[i]
    Fi = zero(Xi) 
    for j in inrange(kdtree, Xi, r)
        F_ij = compute_force(Xi, X[j], V, r²)
        Fi += F_ij
    end
    F[i] = Fi
end

using NearestNeighbors
function compute_forces_kdtree(X)
    F = zeros(SVector{3, Float64}, N)
    r² = r^2
    kdtree = KDTree(X)
    for i in eachindex(X) 
        inner_kdtree(F, X, kdtree, i, r, r²)
    end
    return F
end


@info "SHT"
@time F_sht = compute_forces_sht(X, r)
@time F_sht = compute_forces_sht(X, r)

@info "SHT (parallel)"
@time F_sht_p = compute_forces_sht_parallel(X, r)
@time F_sht_p = compute_forces_sht_parallel(X, r)

@info "KD"
@time F_KD = compute_forces_kdtree(X)
@time F_KD = compute_forces_kdtree(X)

using LinearAlgebra
@assert sum(norm, F_sht_p) ≈ sum(norm, F_KD)