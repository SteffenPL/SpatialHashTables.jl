module SpatialHashTables
using StaticArrays

abstract type AbstractSpatialHashTable end

struct BoundedHashTable{Dim,VT<:AbstractVector,FT<:AbstractFloat} <: AbstractSpatialHashTable
    cellcount::VT
    particlemap::VT

    domain_min::SVector{Dim,FT}
    domain_max::SVector{Dim,FT}

    scaledgrid::SVector{Dim,FT}

    linear_indices::LinearIndices{Dim,NTuple{Dim,Base.OneTo{Int64}}}
    cartesian_indices::CartesianIndices{Dim,NTuple{Dim,Base.OneTo{Int64}}}
end

struct SpatialHashTable{Dim,VT<:AbstractVector,FT<:AbstractFloat} <: AbstractSpatialHashTable
    cellcount::VT
    particlemap::VT

    tablesize::Int64
    scaledgrid::SVector{Dim,FT}
    pseudorandom_factors::SVector{Dim,Int64}

    linear_indices::LinearIndices{Dim,NTuple{Dim,Base.OneTo{Int64}}}
    cartesian_indices::CartesianIndices{Dim,NTuple{Dim,Base.OneTo{Int64}}}
end

function dimension(ht::AbstractSpatialHashTable)
    return length(ht.scaledgrid::SVector{Dim,FT})
end

function boxindex(ht::BoundedHashTable, pos)
    x = ceil.(Int64, @. (pos - ht.domain_min) * ht.scaledgrid)
    return ht.linear_indices[x...]
end

function boxindex(ht::SpatialHashTable, pos)
    x = ceil.(Int64, @. (pos - ht.domain_min) * ht.scaledgrid)
    return mod(reduce(⊻, x * ht.pseudorandom_factors), ht.tablesize)
end

function BoundedHashTable(domain_min::SVector, domain_max, grid, n_positions::Integer)
    cellcount = Vector{Int64}(undef, prod(grid) + 1)
    particlemap = Vector{Int64}(undef, n_positions)

    scaledgrid = grid ./ (domain_max - domain_min)

    linear_indices = LinearIndices(grid)
    cartesian_indices = CartesianIndices(grid)

    return BoundedHashTable(cellcount, particlemap, domain_min, domain_max, grid, scaledgrid, linear_indices, cartesian_indices)
end

BoundedHashTable(domain_max::SVector, grid, np::Integer) = BoundedHashTable(zero(domain_max), domain_max, grid, np)

function BoundedHashTable(X::AbstractVector, args...)
    ht = BoundedHashTable(args..., length(X))
    updateboxes!(ht, X)
    return ht
end

function SpatialHashTable(tablesize::Int64, scaledgrid, n_positions::Integer, pseudorandomfactors = [92837111, 689287499, 283923481])
    cellcount = Vector{Int64}(undef, tablesize + 1)
    particlemap = Vector{Int64}(undef, n_positions)

    linear_indices = LinearIndices(grid)
    cartesian_indices = CartesianIndices(grid)

    pseudorandomfactors = pseudorandomfactors[1:length(scaledgrid)]

    return SpatialHashTable(cellcount, particlemap, domain_min, domain_max, grid, scaledgrid, linear_indices, cartesian_indices)
end



function updateboxes!(ht::SpatialHashTable, X)
    ht.cellcount .= 0
    for i in eachindex(X)
        ht.cellcount[boxindex(ht, X[i])] += 1
    end
    cumsum!(ht.cellcount, ht.cellcount)

    for i in eachindex(X)
        box = boxindex(ht, X[i])
        ht.particlemap[ht.cellcount[box]] = i
        ht.cellcount[box] -= 1
    end
end

function Base.resize!(ht::SpatialHashTable, n_positions)
    resize!(ht.particlemap, n_positions)
end

@inline function iterate_box(ht::SpatialHashTable, box_index)
    box_start = ht.cellcount[box_index] + 1
    box_end = ht.cellcount[box_index+1]
    return (ht.particlemap[k] for k in box_start:box_end)
end

@inline function neighbouring_boxes(ht::SpatialHashTable, box_index, r)
    width = ceil(Int64, r * maximum(ht.scaledgrid))
    offsets = CartesianIndices(ntuple(i -> -width:width, dimension(ht)))
    box = ht.cartesian_indices[box_index]
    return (ht.linear_indices[box+offset] for offset in offsets if checkbounds(Bool, ht.linear_indices, box + offset))
end

@inline function neighbours(ht::SpatialHashTable, pos, r)
    box = boxindex(ht, pos)
    return (k for bj in neighbouring_boxes(ht, box, r) for k in iterate_box(ht, bj))
end

export SpatialHashTable, updateboxes!, resize!, neighbours, boxindex, neighbouring_boxes, iterate_box
end
