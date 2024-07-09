using Revise

using SpatialHashTables
using StaticArrays
using CUDA 
using KernelAbstractions
const KA = KernelAbstractions
using LinearAlgebra
using BenchmarkTools
using KernelAbstractions: @context

N = 10_000
X = 10 .* rand(SVector{2,Float32}, N)
F = similar(X)
F2 = similar(X)

gr = HashGrid(0.1, 100*100, X)

function native(X, grid)

    count = 0

    for i in eachindex(X)
        Xi = X[i]
        for j in neighbours(grid, Xi, 0.1)
            d = sqrt(sum( z -> z^2, X[i] - X[j]))
            if d < 0.1 
                count += 1 
            end
        end
    end
    return count
end

@time native(X, grid)


using SpatialHashTables: compute_interactions!

fnc(i,j,Xi,Xj,Xij,d²) = Xij / sqrt(d²)
@time compute_interactions!(fnc, F, X, gr, 0.1)
@profview compute_interactions!(fnc, F, X, grid, 0.1)

E = zeros(length(X))
@btime compute_interactions!( $((i, j, Xi, Xj, Xij, d²) -> 1.0 / sqrt(d²)), E, X, grid, 0.1)


grid_ = BoundedGrid(0.1, (100, 100), X)
@time compute_interactions!(fnc, F2, X, grid_, 0.1)


query = neighbours(grid, X[1], 0.1)

closeby(X::Vector, i, r) = [j for j in eachindex(X) if norm(X[j] - X[i]) < r]
closeby(grid, i, r) = [j for j in HashGridQuery(grid, X[i], r)]

A = closeby(X, 2, 0.1)
B = closeby(grid, 2, 0.1)
intersect(A,B)

@kernel function interaction_kernel!(F, X, grid)
    i = @index(Global)

    hashes = SpatialHashTables.init_hashes(@context, grid)

    Xi = X[i]
    Fi = zero(eltype(X))
    for j in neighbours(grid, X[i], 0.1, hashes)
        Xj = X[j]
        Xij = Xi - Xj 
        d² = sum(x -> x^2, Xij)
        if i != j && d² < 0.01
            Fi += Xij ./ sqrt(d²)
        end
    end
    F[i] = Xi
end


interaction_kernel!(CPU(), 64)(F, X, grid; ndrange = length(X))
F

X_ = cu(X)
F_ = cu(F)

grid_ = HashGrid(0.1f0, 100, X_, CuVector{Int32})
updatecells!(grid_, X_)


@time compute_interactions!(fnc, F_, X_, grid_, 0.1f0)
