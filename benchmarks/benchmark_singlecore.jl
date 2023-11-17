using StaticArrays 
using SpatialHashTables 
using CellListMap



using CellListMap.PeriodicSystems
N = 100000
const r = 5/sqrt(N)

system = PeriodicSystem(
           xpositions = [ r .+ x*(1-2*r) for x in rand(SVector{3,Float64},N)], 
           unitcell=[1.0,1.0,1.0], 
           cutoff = r, 
           output = 0.0,
           output_name = :energy
       );


function energy(d2, u) 
    if d2 < r^2
        u += atan(d2) 
    end
    return u 
end


function map_energy(x,y,i,j,d2,u) 
    energy(d2, u)
end

@inline interation_pairs(ht, X, cutoff) = ( (i,j) for i in eachindex(X) for j in neighbours(ht, X[i], cutoff) )
function test_speed(st, X, cutoff)
    e = 0.0
    N = length(X)
    for (i, j) in interation_pairs(st, X, cutoff)
        if i < j
            d2 = sum( x -> x^2, X[i] - X[j])
            if d2 < cutoff^2
                e += atan(d2)
            end
        end
    end
    e
end

# warm up
CellListMap.PeriodicSystems.map_pairwise!(map_energy, system)
r1 = CellListMap.PeriodicSystems.map_pairwise!(map_energy, system; update_lists=false)

domain = (min = @SVector[0.0,0.0,0.0], max = @SVector[1.0,1.0,1.0])
grid = tuple( ceil.(Int64, (domain.max - domain.min) / system.cutoff / 2 )... )
X = system.xpositions
N = length(X)

st = SpatialHashTable( domain , grid, N)
updateboxes!(st, X)

# warm up
r2 = test_speed(st, X, system.cutoff)


@assert r1 ≈ r2

using BenchmarkTools
@btime test_speed($st, $X, $(system.cutoff))
@btime CellListMap.PeriodicSystems.map_pairwise!($map_energy, $system; update_lists=false)
