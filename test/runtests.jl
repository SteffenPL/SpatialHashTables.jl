using SpatialHashTables
using StaticArrays
using Test

using SpatialHashTables: hashindex, gridindices

const SVec2 = SVector{2, Float64}

@testset "SpatialHashTables.jl" begin
        
    N = 1000
    X = rand(SVec2, N)
    cutoff = 1/sqrt(N)
    cellsize = cutoff
    
    range = [1.0, 2.0]
    ht = BoundedHashTable(X, cellsize, range)

    @test dimension(ht) == 2
    @test hashindex(ht, (1, 1)) == 1
    @test hashindex(ht, (2, 1)) == 2

    # @test_throws BoundsError hashindex(ht, (0,0))

    updateboxes!(ht, X)

    nb = collect(neighbours(ht, X[1], 0.1))
    real_nb = [i for i in eachindex(X) if sqrt(sum( x -> x^2, X[i] - X[1])) < 0.1]

    @test issubset(real_nb, nb)

    cellsize = SVec2(0.1,0.1)
    tablesize = 100
    sht = SpatialHashTable(length(X), tablesize, cellsize)


    updateboxes!(sht, X)

    nb = collect(neighbours(sht, X[1], 0.1))
    real_nb = [i for i in eachindex(X) if sqrt(sum( x -> x^2, X[i] - X[1])) < 0.1]

    @test issubset(real_nb, nb)
end

@testset "BoundedHashTable" begin 
    BoundedHashTable(10, (2,2), [1.0, 1.0])
    BoundedHashTable(10, (2,2), [1.0, 1.0], [2.0, 2.0])
    BoundedHashTable(10, 1.0, [1.0, 1.0])

    X = rand(SVec2, 10)
    BoundedHashTable(X, (2,2), [1.0, 1.0])
    BoundedHashTable(X, (2,2), [0.0, 0.0], [2.0, 2.0])
    BoundedHashTable(X, 1.0, [1.0, 1.0])
end

@testset "SpatialHashTable" begin 
    SpatialHashTable(10, 5, (0.5,0.5))
    SpatialHashTable(10, 5, 0.5)
    SpatialHashTable(10, 1, [1.0, 1.0])

    X = rand(SVec2, 10)
    SpatialHashTable(X, 5, (0.5,0.5))
    SpatialHashTable(X, 5, 0.5)
    SpatialHashTable(X, 1, [1.0, 1.0])
end

@testset "Hash index collisions" begin 
    X = rand(SVec2, 100)
    ht = SpatialHashTable(X, 2, 0.1)

    @test allunique(neighbours(ht, X[1], 0.1))
end