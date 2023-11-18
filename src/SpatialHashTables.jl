module SpatialHashTables
using StaticArrays

abstract type AbstractSpatialHashTable end


struct BoundedHashTable{Dim,VT<:AbstractVector,FT<:AbstractFloat} <: AbstractSpatialHashTable
    cellcount::VT
    particlemap::VT

    domainstart::SVector{Dim,FT}
    domainend::SVector{Dim,FT}

    inv_cellsize::SVector{Dim,FT}

    linear_indices::LinearIndices{Dim,NTuple{Dim,Base.OneTo{Int64}}}
    gridsize::NTuple{Dim,Int64} # CartesianIndices{Dim,NTuple{Dim,Base.OneTo{Int64}}}
end

struct SpatialHashTable{Dim,VT<:AbstractVector,FT<:AbstractFloat} <: AbstractSpatialHashTable
    cellcount::VT
    particlemap::VT

    tablesize::Int64
    inv_cellsize::SVector{Dim,FT}

    pseudorandom_factors::SVector{Dim,Int64}
end

dimension(::SpatialHashTable{Dim}) where {Dim} = Dim
dimension(::BoundedHashTable{Dim}) where {Dim} = Dim


function BoundedHashTable(N::Int64, grid::Tuple, domainstart::SVector, domainend::SVector)
    
    cellcount = Vector{Int64}(undef, prod(grid) + 1)
    particlemap = Vector{Int64}(undef, N)

    inv_cellsize = grid ./ (domainend - domainstart)

    linear_indices = LinearIndices(grid)
    #cartesian_indices = CartesianIndices(grid)

    return BoundedHashTable(cellcount, particlemap, domainstart, domainend, inv_cellsize, linear_indices, grid)
end

function BoundedHashTable(N::Int64, grid, domainstart, domainend)
    domainstart = SVector{length(grid),Float64}(domainstart)
    domainend = SVector{length(grid),Float64}(domainend)
    return BoundedHashTable(N, grid, domainstart, domainend)
end

BoundedHashTable(N::Int64, grid, range) = BoundedHashTable(N, grid, zero(range), range)

function BoundedHashTable(N::Int64, cellsize::Number, domainstart::SVector, domainend::SVector)
    grid = @. max(1, floor(Int64, (domainend - domainstart) / cellsize))
    return BoundedHashTable(N, tuple(grid...), domainstart, domainend)
end


function BoundedHashTable(N::Int64, cellsize::Number, domainstart, domainend)
    domainstart = SVector{length(domainstart),Float64}(domainstart)
    domainend = SVector{length(domainstart),Float64}(domainend)
    return BoundedHashTable(N, cellsize, domainstart, domainend)
end

function BoundedHashTable(X::AbstractVector, args...)
    ht = BoundedHashTable(length(X), args...)
    updateboxes!(ht, X)
    return ht
end


function SpatialHashTable(N::Int64, tablesize::Int64, cellsize::SVector, pseudorandomfactors = (92837111, 689287499, 283923481))
    cellcount = Vector{Int64}(undef, tablesize + 1)
    particlemap = Vector{Int64}(undef, N)

    inv_cellsize = 1 ./ cellsize

    Dim = length(cellsize)
    pseudorandomfactors = SVector{Dim,Int64}(pseudorandomfactors[i] for i in eachindex(inv_cellsize))

    return SpatialHashTable(cellcount, particlemap, tablesize, inv_cellsize, pseudorandomfactors)
end

SpatialHashTable(N::Int64, tablesize::Int64, cellsize, args...) = SpatialHashTable(N, tablesize, SVector{length(cellsize),Float64}(cellsize), args...)

function SpatialHashTable(N::Int64, tablesize::Int64, cutoff::Number, Dim::Int64, args...)
    return SpatialHashTable(N, tablesize, fill(cutoff, Dim), args...)
end

function SpatialHashTable(X::AbstractVector, tablesize::Int64, cutoff::Number, args...)
    Dim = length(X[1])
    ht = SpatialHashTable(length(X), tablesize, cutoff, Dim, args...)
    updateboxes!(ht, X)
    return ht
end

function SpatialHashTable(X::AbstractVector, args...)
    ht = SpatialHashTable(length(X), args...)
    updateboxes!(ht, X)
    return ht
end




function updateboxes!(ht::AbstractSpatialHashTable, X)
    ht.cellcount .= 0
    for i in eachindex(X)
        ht.cellcount[hashposition(ht, X[i])] += 1
    end
    cumsum!(ht.cellcount, ht.cellcount)

    for i in eachindex(X)
        box = hashposition(ht, X[i])
        ht.particlemap[ht.cellcount[box]] = i
        ht.cellcount[box] -= 1
    end
end

function Base.resize!(ht::AbstractSpatialHashTable, n_positions)
    resize!(ht.particlemap, n_positions)
end





insidegrid(ht::BoundedHashTable, gridpos) = all(@. 1 <= gridpos <= ht.gridsize)
gridindices(ht::BoundedHashTable, pos) = ceil.(Int64, (pos - ht.domainstart) .* ht.inv_cellsize)
hashindex(ht::BoundedHashTable, gridindices) = ht.linear_indices[gridindices...]

insidegrid(::SpatialHashTable, gridpos) = true
gridindices(ht::SpatialHashTable, pos) = ceil.(Int64, @. pos * ht.inv_cellsize)

function hashindex(ht::SpatialHashTable, gridindices)
    return 1 + mod(abs(reduce(⊻, gridindices .* ht.pseudorandom_factors)), ht.tablesize)
end

hashposition(ht::AbstractSpatialHashTable, pos) = hashindex(ht, gridindices(ht, pos))



function iterate_box(ht::AbstractSpatialHashTable, boxhash)
    box_start = ht.cellcount[boxhash] + 1
    box_end = ht.cellcount[boxhash+1]
    return (ht.particlemap[k] for k in box_start:box_end)
end

function neighbouring_boxes(ht::AbstractSpatialHashTable, gridpos, r)
    Dim = dimension(ht)
    widths = @. ceil(Int64, r * ht.inv_cellsize)
    
    int_offsets = CartesianIndices(ntuple(i -> -widths[i]:widths[i], Dim))
    offsets = (gridpos .+ Tuple(i) for i in int_offsets)
    return (hashindex(ht, offset) for offset in offsets if insidegrid(ht, offset))
end

function neighbours(ht::AbstractSpatialHashTable, pos, r)
    gridpos = gridindices(ht, pos)
    return (k for boxhash in neighbouring_boxes(ht, gridpos, r) for k in iterate_box(ht, boxhash))
end

export SpatialHashTable, BoundedHashTable, AbstractSpatialHashTable, updateboxes!, resize!, neighbours, iterate_box, dimension, inside 
end
