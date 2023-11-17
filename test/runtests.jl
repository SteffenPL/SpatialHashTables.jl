using SpatialHashTables
using StaticArrays
using Test

const SVec2 = SVector{2, Float64}

@testset "SpatialHashTables.jl" begin

    
    X = rand(SVec2, 100)
    domain = (min = SVec2(0, 0), max = SVec2(1, 1))
    grid = (5, 5)

    ht = SpatialHashTable(domain, grid, length(X))
    updateboxes!(ht, X)

    X = rand(SVec2, 1000)
    domain = (min = SVec2(0, 0), max = SVec2(1, 1))
    grid = (5, 5)

    resize!(ht, length(X))
    updateboxes!(ht, X)

    nb = collect(neighbours(ht, X[1], 0.1))
    real_nb = [i for i in eachindex(X) if sqrt(sum( x -> x^2, X[i] - X[1])) < 0.1]

    @test issubset(real_nb, nb)
end


#= 

function get_speed(ht, N)
    X = rand(SVec2, N)
    resize!(ht, length(X))
    updateboxes!(ht, X)

    return @elapsed( sum( X[i][1] - X[j][1] for i in eachindex(X) for j in neighbours(ht, X[i], 0.1) ) )
end

Ns = round.(Int64, 2 .^ (1:14))

s = [get_speed(ht, N) for N in Ns]


s ./ Ns

=#