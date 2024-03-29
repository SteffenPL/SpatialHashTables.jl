using CellListMap.PeriodicSystems
using CellListMap.PeriodicSystems: map_pairwise!

dist_sq(a,b) = sum(x -> x*x, a - b)

function setup(N, Dim = 3, r = 1/(N)^(1/Dim); dtype = Float64)

    r = dtype(r)
    X = rand(SVector{Dim,dtype},N)
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
