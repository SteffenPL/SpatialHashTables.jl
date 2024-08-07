using SpatialHashTables
using StaticArrays
using Test, LinearAlgebra

const SHT = SpatialHashTables
const SVec2 = SVector{2, Float64}


@testset "elementary operations (BoundedGrid)" begin 
#begin
    r = 0.25
    grid = BoundedGrid(r, (4,4), 3)
    X = SVec2.([(0.9,0.9), (0.3,0.6), (0.45, 0.7)])

    @test SHT.pos2grid(grid, X[1]) == (4, 4)
    @test SHT.pos2grid(grid, X[2]) == (2, 3)
    @test SHT.pos2hash(grid, X[1]) == 16
    @test SHT.pos2hash(grid, X[2]) == 10
    @test SHT.pos2hash(grid, X[2] .+ SVec2(1.0, 1.0)) == 10

    # @test_throws BoundsError SHT.grid2hash(grid, (5,5))

    updatecells!(grid, X)
    @test grid[4,4] == [1]
    @test grid[2,3] == [2,3]
    @test SHT.celldomain(grid, (2,2)) == (SVec2(0.25, 0.25), SVec2(0.5, 0.5))

    #cellcenter = 0.5*(sum(SHT.celldomain(grid, 5)))
    #@test grid[5] == collect(HashGridQuery(grid, cellcenter, r/3))

    #cellcenter = 0.5*(sum(SHT.celldomain(grid, 10)))
    #@test grid[10] == collect(HashGridQuery(grid, cellcenter, r/3))
end


@testset "random points" begin
    N = 500
    X = rand(SVec2, N)

    push!(X, SVec2(0.95, 0.5))
    N += 1 

    cutoff = 1/sqrt(N)
    cellsize = cutoff

    grid = BoundedGrid(SVec2(1,1) ./ 10, (10, 10), X)

    function naivecell(X, a, b)
        return [i for i in eachindex(X) if all(a .< X[i] .<= b)]
    end

    @test naivecell(X, 0.4, 0.5) == grid[5,5]
    # @test naivecell(X, SHT.celldomain(grid, 3)...) == grid[3]


    @test naivecell(X, 0.3, 0.7) == sort(collect(neighbours(grid, SVec2(0.5, 0.5), 0.15)))
    @test naivecell(X, 0.4, 0.5) == sort(collect(neighbours(grid, SVec2(0.45, 0.45), 0.02)))
    @test naivecell(X, 0.0, 1.0) == sort(collect(neighbours(grid, SVec2(0.5, 0.5), 1.0)))

    # @test all(naivecell(X, SHT.celldomain(grid, ci)...) == sort(grid[ci]) for ci in SHT.cartesianindices(grid) )

    x1 = 0.0
    for i in 1:N
        for j in 1:N 
            d = norm(X[i] - X[j])
            if d < 0.1
                x1 += dot(X[i], X[j])
            end
        end
    end

    x2 = 0.0
    for i in 1:N
        for j in neighbours(grid, X[i], 1.0)
            d = norm(X[i] - X[j])
            if d < 0.1
                x2 += dot(X[i], X[j])
            end
        end
    end
    
    @test x1 ≈ x2


    # periodic boundary conditions 
    @test N in collect(neighbours(grid, SVec2(0.01, 0.5), 0.1))
    @test !(N in collect(neighbours(grid, SVec2(0.01, 0.3), 0.1)))
end 


@testset "Subset positions" begin 
    X = [SA[0.1, 0.1, 0.5], SA[0.1, 0.1, 0.0]]
    grid = BoundedGrid(0.1, (10, 10), X)
    @test grid[1,1] == [1,2]


    X = [SA[0.1, 0.5, 0.1], SA[0.1, 0.0, 0.1]]
    grid = BoundedGrid(0.1, (10, 10), X)
    @test grid[1,5] == [1]


    X = [SA[0.1, 0.1, 0.5], SA[0.1, 0.1, 0.0]]
    grid = HashGrid(0.1, 100, X; inds = SOneTo(2))
    @test grid[1,1] == [1,2]


    X = [SA[0.1, 0.5, 0.1], SA[0.1, 0.0, 0.1]]
    grid = HashGrid(0.1, 100, X)
    @test grid[1,5,1] == [1]
end