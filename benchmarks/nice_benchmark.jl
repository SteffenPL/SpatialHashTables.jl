using Revise
using StaticArrays 
using LinearAlgebra

using SpatialHashTables

const SVec3d = SVector{3,Float64}
const SVec3f = SVector{3,Float32}

function example_set(N)
    N = 10_000
    r = 1/N^(1/3)
    X = rand(SVector{3, Float64}, N)

    return X, r, (@SVector[0.0,0.0,0.0], @SVector[1.0,1.0,1.0])
end


X, r, bounds = example_set(1000)
F = similar(X)
gridsize = Tuple(@. ceil(Int, (bounds[2] - bounds[1]) / r))

grid = HashGrid(X, bounds..., gridsize)
grids = HashGrid(X, bounds..., (1,1,1))

fnc(Xi, Xj, d) = (Xi - Xj) / d

function test_naive!(F, X, r)
    for i in eachindex(X)
        F[i] = zero(SVec3d)
        for j in eachindex(X)
            Xij = X[i] - X[j]
            d = norm(Xij)
            if 0 < d < r
                F[i] += fnc(X[i], X[j], d)
            end
        end
    end
    return nothing
end
LinearIndices
function test_singlecore!(F, X, r, grid)
    for i in eachindex(X)
        F[i] = zero(SVec3d)
        for j in HashGridQuery(grid, X[i], r)
            Xij = X[i] - X[j]
            d = norm(Xij)
            if 0 < d < r
                F[i] += fnc(X[i], X[j], d)
            end
        end
    end
    return nothing
end

function test_singlecore_manual!(F, X, r, grid)
    @inbounds for i in eachindex(X)
        F[i] = zero(SVec3d)
        hg = HashGridQuery(grid, X[i], r)
        (k, s) = iterate(hg)
        F[i] += 10.0 * k * F[i]
        for j in eachindex(X)
            Xij = X[i] - X[j]
            d = norm(Xij)
            if 0 < d < r
                F[i] += fnc(X[i], X[j], d)
            end
        end
    end
    return nothing
end

Fn = similar(F)
Fs = similar(F)

@time test_naive!(Fn, X, r)
@time test_singlecore!(Fs, X, r, grid)
@time test_singlecore!(Fs, X, r, grids)
@time test_singlecore_manual!(Fs, X, r, grid)


# @profview test_naive!(Fn, X, r)
# @profview test_singlecore!(Fs, X, r, grid)

# @code_llvm test_singlecore!(Fs, X, r, grid)

# hg = HashGridQuery(grid, X[2], r)

