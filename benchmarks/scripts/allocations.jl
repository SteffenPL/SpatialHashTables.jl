using Revise
using StaticArrays 
using SpatialHashTables 
using CellListMap


using SpatialHashTables: neighbouring_boxes, iterate_box, hashindex, gridindices, boxindex
include("setup.jl")  # defines: N, X,

N, r, X, energy = setup(10_000)

ht = BoundedHashTable(X, r, [1.0, 1.0, 1.0])

gridpos = gridindices(ht, X[1])
Dim = dimension(ht)

boxhash = hashindex(ht, gridpos)
collect(iterate_box(ht, boxhash))

function test_a(ht, X, r)
    r = 0
    for x in X
    gp = gridindices(ht, x)
    nbx = neighbouring_boxes(ht, gp, r)
    for nb in nbx
        r += 1
    end
end 
    return r 
end
@profview test_a(ht, X, r)


x = X[3]

gp = gridindices(ht, x)
nbx = neighbouring_boxes(ht, gp, r)

@code_warntype neighbouring_boxes(ht, gp, 0.01)
collect(nbx)

Dim = dimension(ht)

widths = @. ceil(Int64, r * ht.inv_cellsize)
ntuple(i -> -widths[i]:widths[i], Dim)

int_offsets = Iterators.product(ntuple(i -> -widths[i]:widths[i], Dim)...)



@code_w 

domain = (min = @SVector[0.0, 0.0, 0.0], max = @SVector[1.0, 1.0, 1.0] )

N = 100000
X = [rand(SVector{3,Float64}) for i in 1:N]

r = 1/sqrt(N)


# SpatialHashTables
grid = (100, 100, 100)
st = SpatialHashTable(domain, grid, N)

updatetable!(st, X)

function test_speed(st, X, r)
    mean_distance = SVector{3,Float64}(0.,0.,0.0)
    cutoff = r
    N = length(X)
    for i in 1:N 
        for j in neighbours(st, X[i], r)
            xixj = X[i] - X[j]
            d = sum(x -> x^2, xixj)
            if d < cutoff^2
                mean_distance += xixj ./ sqrt(d)
            end
        end
    end
    mean_distance
end

function test_speed_naive(st, X, r)
    mean_distance = SVector{3,Float64}(0.,0.,0.0)
    cutoff = r
    N = length(X)
    for i in 1:N 
        for j in 1:i-1
            xixj = X[i] - X[j]
            d2 = sum(x -> x^2, xixj)
            if d2 < cutoff^2
                mean_distance += xixj ./ sqrt(d2)
            end
        end
    end
    mean_distance
end


@time test_speed(st, X, r)
@time test_speed_naive(st, X, r)

function energy(d2, u) 
    if d2 < 0.1^2
        u += 1 / sqrt(d2) 
    end
    return u 
end


using CellListMap.PeriodicSystems
N = 100000
system = PeriodicSystem(
           xpositions = rand(SVector{3,Float64},N), 
           unitcell=[1.0,1.0,1.0], 
           cutoff = r, 
           output = 0.0,
           output_name = :energy
       );

using BenchmarkTools
@btime CellListMap.PeriodicSystems.map_pairwise!( $map_energy, $system; update_lists=false)


map_energy(x,y,i,j,d2,u) = energy(d2, u)
@time CellListMap.PeriodicSystems.map_pairwise!(map_energy, system)

@inline interation_pairs(ht, X, cutoff) = ( (i,j) for i in eachindex(X) for j in neighbours(ht, X[i], cutoff) )
function test_speed(st, X, cutoff)
    e = 0.0
    N = length(X)
    for (i, j) in interation_pairs(st, X, cutoff)
        if i < j
            d2 = sum( x -> x^2, X[i] - X[j])
            if d2 < cutoff^2
                e += 1 / sqrt(d2)
            end
        end
    end
    e
end

function test_ref(X, cutoff)
    e = 0.0
    N = length(X)
    for i in 1:N, j in 1:N
        if i < j
            d2 = sum( x -> x^2, X[i] - X[j])
            if d2 < cutoff^2
                e += 1 / sqrt(d2)
            end
        end
    end
    e
end



grid = tuple( ceil.(Int64, (domain.max - domain.min) / system.cutoff / 2 )... )
domain = (min = @SVector[0.0,0.0,0.0], max = @SVector[1.0,1.0,1.0])
X = system.xpositions
N = length(X)

st = SpatialHashTable( domain , grid, N)
updatetable!(st, X)

@time test_speed(st, X, system.cutoff)
@time CellListMap.PeriodicSystems.map_pairwise!(map_energy, system; update_lists=false)

@btime test_speed($st, $X, $(system.cutoff))
@time test_ref(X, system.cutoff)

@profview CellListMap.PeriodicSystems.map_pairwise!(map_energy, system; update_lists=false)
@profview test_speed(st, X, system.cutoff)



system = PeriodicSystem(
           xpositions = map( x -> x .+ 0.1, X), 
           unitcell=[1.2,1.2,1.2], 
           cutoff = 0.1, 
           output = 0.0,
           output_name = :energy
       );

CellListMap.PeriodicSystems.map_pairwise!( map_energy, system )



@time test_speed(st, X, system.cutoff)
@time CellListMap.PeriodicSystems.map_pairwise!( map_energy, system )

@btime CellListMap.PeriodicSystems.map_pairwise!( $((x,y,i,j,d2,u) -> energy(d2, u)), $system; update_lists=false)

@btime test_speed($st, $X, $(system.cutoff))






using BenchmarkTools


X = rand(SVector{2, Float64}, 100)
domain = (min = SVector{2, Float64}(0, 0), max = SVector{2, Float64}(1, 1))
grid = (5, 5)

ht = SpatialHashTable(domain, grid, X)

# the structure can also be resized and updated 
X2 = rand(SVector{2, Float64}, 1000)
resize!(ht, length(X2))
updatetable!(ht, X2)

interations(ht, X, R) = ( (i,j) for i in eachindex(X) for j in neighbours(ht, X[i], R) )
function test_allocations(ht, X, R)
    F = @SVector [0.0, 0.0]
    for (i,j) in interations(ht, X, R)
        if i < j
            F += X[i] - X[j]
        end
    end
    return F
end

@time test_allocations(ht, X2, 0.1)