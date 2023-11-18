module SpatialHashTables
using StaticArrays
import Adapt: adapt_structure

abstract type AbstractSpatialHashTable end


struct BoundedHashTable{Dim,VT<:AbstractVector,FT<:AbstractFloat,IT<:Integer} <: AbstractSpatialHashTable
    cellcount::VT
    particlemap::VT

    domainstart::SVector{Dim,FT}
    domainend::SVector{Dim,FT}

    inv_cellsize::SVector{Dim,FT}

    strides::NTuple{Dim,IT}
    gridsize::NTuple{Dim,IT}
end

struct SpatialHashTable{Dim,VT<:AbstractVector,FT<:AbstractFloat} <: AbstractSpatialHashTable
    cellcount::VT
    particlemap::VT

    tablesize::Int64
    inv_cellsize::SVector{Dim,FT}

    pseudorandom_factors::SVector{Dim,Int64}

    caches::Vector{Set{Int64}}
end

dimension(::SpatialHashTable{Dim}) where {Dim} = Dim
dimension(::BoundedHashTable{Dim}) where {Dim} = Dim

inttype(ht::AbstractSpatialHashTable) = eltype(ht.cellcount)

function BoundedHashTable(N::Integer, grid::Tuple, domainstart::SVector, domainend::SVector)
    
    cellcount = Vector{typeof(N)}(undef, prod(grid) + 1)
    particlemap = Vector{typeof(N)}(undef, N)

    inv_cellsize = grid ./ (domainend - domainstart)
    strides = (oneunit(eltype(grid)), cumprod(grid[1:end-1])...)

    return BoundedHashTable(cellcount, particlemap, domainstart, domainend, inv_cellsize, strides, grid)
end

function BoundedHashTable(N::Integer, grid, domainstart, domainend)
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

const default_factors = (92837111, 689287499, 283923481)

function SpatialHashTable(N::Int64, tablesize::Int64, cellsize::SVector; cachesize = Threads.nthreads(),  pseudorandomfactors = default_factors)
    cellcount = Vector{Int64}(undef, tablesize + 2) # we need one extra for Julia type stability...
    particlemap = Vector{Int64}(undef, N)

    inv_cellsize = 1 ./ cellsize

    Dim = length(cellsize)
    pseudorandomfactors = SVector{Dim,Int64}(pseudorandomfactors[i] for i in eachindex(inv_cellsize))

    caches = [ Set{Int64}() for i in 1:cachesize ]

    return SpatialHashTable(cellcount, particlemap, tablesize, inv_cellsize, pseudorandomfactors, caches)
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
gridindices(ht::BoundedHashTable, pos) = ceil.(inttype(ht), (pos - ht.domainstart) .* ht.inv_cellsize)
hashindex(ht::BoundedHashTable, gridindices) = sum( @. (gridindices-1) * ht.strides ) + 1

insidegrid(::SpatialHashTable, gridpos) = true
gridindices(ht::SpatialHashTable, pos) = ceil.(inttype(ht), @. pos * ht.inv_cellsize)

function hashindex(ht::SpatialHashTable, gridindices)
    return oneunit(inttype(ht)) + mod(abs(reduce(⊻, gridindices .* ht.pseudorandom_factors)), ht.tablesize)
end

hashposition(ht::AbstractSpatialHashTable, pos) = hashindex(ht, gridindices(ht, pos))



function iterate_box(ht::AbstractSpatialHashTable, boxhash)
    box_start = ht.cellcount[boxhash] + oneunit(inttype(ht))
    box_end = ht.cellcount[boxhash+1]
    return (ht.particlemap[k] for k in box_start:box_end)
end

function neighbouring_boxes(ht::AbstractSpatialHashTable, gridpos, r)
    IT = inttype(ht)
    Dim = IT(dimension(ht))
    widths = @. ceil(IT, r * ht.inv_cellsize)
    int_offsets = CartesianIndices(ntuple(i -> -widths[i]:widths[i], Dim))
    offsets = (gridpos .+ Tuple(i) for i in int_offsets)
    return (hashindex(ht, offset) for offset in offsets if insidegrid(ht, offset))
end

function neighbours(ht::AbstractSpatialHashTable, pos, r)
    gridpos = gridindices(ht, pos)
    return (k for boxhash in neighbouring_boxes(ht, gridpos, r) for k in iterate_box(ht, boxhash))
end


# The following code deals with hash index collisions which could result 
# in the same index being returned multiple times.
thread_cache(ht::SpatialHashTable) = ht.caches[Threads.threadid()]

function iterate_box_if(ht, boxhash, seen)
    if boxhash in seen
        return iterate_box(ht, ht.tablesize+1)  # this box is empty (see constructor where we add +2)
    else
        push!(seen, boxhash)
        return iterate_box(ht, boxhash)
    end
end

function neighbours(ht::SpatialHashTable, pos, r)
    gridpos = gridindices(ht, pos)
    seen = thread_cache(ht)
    empty!(seen)
    return (k for boxhash in neighbouring_boxes(ht, gridpos, r) for k in iterate_box_if(ht, boxhash, seen))
end


# GPU support for BoundedHashTable
function adapt_structure(to, bht::BoundedHashTable)
    cellcount = adapt_structure(to, Int32.(bht.cellcount))
    particlemap = adapt_structure(to, Int32.(bht.particlemap))
    return BoundedHashTable(cellcount, 
                            particlemap,
                            Float32.(bht.domainstart), 
                            Float32.(bht.domainend), 
                            Float32.(bht.inv_cellsize), 
                            Int32.(bht.strides), 
                            Int32.(bht.gridsize))
end


export SpatialHashTable, BoundedHashTable, AbstractSpatialHashTable, updateboxes!, resize!, neighbours, iterate_box, dimension, inside 
end
