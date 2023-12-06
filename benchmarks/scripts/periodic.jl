using Revise
using SpatialHashTables
using StaticArrays
using LinearAlgebra
using Test

using SpatialHashTables: hashindex, gridindices

const SVec2 = SVector{2, Float64}

X = [SVec2(0.01,0.01), SVec2(0.99, 0.99)]
ht = BoundedHashTable(X, 0.1, [1.0, 1.0])
d = 0.0
for i in eachindex(X)
    for (j, offset) in periodic_neighbours(ht, X[i], 0.1)
        if i != j
            #@show (i,j) offset
            d += norm(X[j] - offset - X[i])
        end
    end
end
@test d ≈ 2 * sqrt(2*0.02^2)