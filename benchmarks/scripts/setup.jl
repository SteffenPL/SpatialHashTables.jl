using CellListMap.PeriodicSystems
using CellListMap.PeriodicSystems: map_pairwise!

dist_sq(a,b) = sum(x -> x^2, a - b)

function setup(N, Dim = 3, r = 1/(N)^(1/Dim))

    X = rand(SVector{Dim,Float64},N)
    X = [clamp.(x, r, 1-r) for x in X]  # avoid periodic boundary 

    return N, Dim, r, X
end

function setup_celllistmap(X, cutoff, map_energy)
    system = PeriodicSystem(
        xpositions = X, 
        unitcell = ones(length(X[1])), 
        cutoff = cutoff, 
        output = 0.0,
        output_name = :energy
    )

    map_pairwise!(map_energy, system)
    return system
end
