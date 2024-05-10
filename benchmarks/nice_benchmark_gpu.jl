using Revise
using StaticArrays 
using LinearAlgebra

using KernelAbstractions
using SpatialHashTables
using SpatialHashTables: numthreads, domainsize

const SVec3d = SVector{3,Float64}
const SVec3f = SVector{3,Float32}

function example_set(N)
    r = 1/N^(1/3)
    X = rand(SVector{3, Float64}, N)

    return X, r, (@SVector[0.0,0.0,0.0], @SVector[1.0,1.0,1.0])
end


fnc(Xi, Xj, Xij, d) = Xij / d

function test_naive!(F, X, r, period = @SVector[1.0, 1.0, 1.0])
    periodinv = 1 ./ period

    for i in eachindex(X)
        Xi = X[i]
        Fi = zero(SVec3d)
        for j in eachindex(X)
            Xj = X[j]
            Xij = wrap(Xi - Xj, period, periodinv)
            d = norm(Xij)
            if 0 < d < r
                Fi += fnc(Xi, Xj, Xij, d)
            end
        end
        F[i] = Fi
    end
    return nothing
end

function wrap(v, period, periodinv = 1 ./ period)
    return @. v - period * round(v * periodinv)
end

function test_singlecore!(F, X, r, grid)
    period = domainsize(grid)
    periodinv = 1 ./ period

    for i in eachindex(X)
        Xi = X[i]
        Fi = zero(SVec3d)
        for j in HashGridQuery(grid, Xi, r)
            Xj = X[j]

            Xij = wrap(Xi - Xj, period, periodinv)
            d = norm(Xij)
            if 0 < d < r
                Fi += fnc(Xi, Xj, Xij, d)
            end
        end
        F[i] = Fi
    end
    return nothing
end

@kernel function test_multithreaded_kernel!(F, @Const(X), r, grid, period, periodinv)
    i = @index(Global)
    
    Xi = X[i]
    Fi = zero(SVec3d)

    query = HashGridQuery(grid, X[i], r)
    for j in query
        Xj = X[j]
        Xij = wrap(Xi - Xj, period, periodinv)
        d = norm(Xij)
        if 0 < d < r
            Fi += fnc(Xi, Xj, Xij, d)
        end 
    end
    F[i] = Fi    
end

function test_multithreaded!(F, X, r, grid)
    period = domainsize(grid)
    periodinv = 1 ./ period

    kernel = test_multithreaded_kernel!(grid.backend, numthreads(grid))
    kernel(F, X, r, grid, period, periodinv, ndrange = length(X))
    synchronize(grid.backend)
    return nothing
end

X, r, bounds = example_set(1_000)
F = similar(X)
gridsize = Tuple(@. ceil(Int, (bounds[2] - bounds[1]) / r))

grid = HashGrid(X, bounds..., gridsize)

Fn = similar(F)
Fs = similar(F)
Fm = similar(F)

@time test_naive!(Fn, X, r)
@time test_singlecore!(Fs, X, r, grid)
@time test_multithreaded!(Fm, X, r, grid)


X, r, bounds = example_set(1_000_000)
F = similar(X)
gridsize = Tuple(@. ceil(Int, (bounds[2] - bounds[1]) / r))

grid = HashGrid(X, bounds..., gridsize; nthreads = 1)

Fs = similar(F)
Fm = similar(F)

@time test_singlecore!(Fs, X, r, grid)
@time test_multithreaded!(Fm, X, r, grid)


using Adapt

X_g = cu(SVec3f.(X))
r_g = Float32(r)
bounds_g = SVec3f.(bounds)
gridsize_g = Int32.(gridsize)

grid_g = HashGrid{CuVector{Int32}}(X_g, bounds_g..., gridsize_g)

Fg = similar(X_g)
@time test_multithreaded!(Fg, X_g, r_g, grid_g)


using CellListMap

box = Box(bounds[2], r)
cl = CellList(X, box)
Fcl = similar(F)



function test_cellmaps(Fout, X, r, box, cl)
    fill!(Fout, zero(SVector{3,Float64}))

    function force_(x,y,i,j,d2,F)
        if 0 < d2 && d2 < 0.01^2
            Fij = fnc(x, y, sqrt(d2))
            F[i] += Fij
            F[j] -= Fij
        end 
        return F
    end

    CellListMap.map_pairwise!( force_, Fout, box, cl)
    return nothing
end

fill!(Fcl, zero(SVec3d))
@time map_pairwise!( (x,y,i,j,d2,F) -> force(x,y,i,j,d2,F,r), Fcl, box, cl)
@time test_cellmaps(Fcl, X, r, box, cl)

using BenchmarkTools
@btime test_cellmaps($F, $X, $r, $box, $cl)
@btime test_multithreaded!($F_g, $X_g, $r_g, $grid_g)
@btime test_multithreaded!($Fm, $X, $r, $grid)

system = ParticleSystem(
    xpositions = X, 
    unitcell = bounds[2],
    cutoff = r, 
    output = similar(F),
    output_name = :forces
)

function update_forces!(x,y,i,j,d2,F)
    if 0 < d2 < 0.01^2
        Fij = fnc(x, y, sqrt(d2))
        F[i] += Fij
        F[j] -= Fij
    end 
    return F
end


# @time map_pairwise!(update_forces!, system)

@btime map_pairwise!($update_forces!, $system)