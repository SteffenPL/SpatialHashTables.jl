module SpatialHashTables
using StaticArrays
import Adapt: adapt_structure

"""
    AbstractSpatialHashTable

Abstract type for spatial hash tables.
"""
abstract type AbstractSpatialHashTable end

"""
    BoundedHashTable

A spatial hash table with a fixed domain.
"""
struct BoundedHashTable{Dim,VT<:AbstractVector,FT<:AbstractFloat,IT<:Integer} <: AbstractSpatialHashTable
    cellcount::VT
    particlemap::VT

    domainstart::SVector{Dim,FT}
    domainend::SVector{Dim,FT}

    inv_cellsize::SVector{Dim,FT}

    strides::NTuple{Dim,IT}
    gridsize::NTuple{Dim,IT}
end

""" 
    SpatialHashTable

A spatial hash table for unbounded domains. See [Matthias Müller's paper](https://matthias-research.github.io/pages/publications/tetraederCollision.pdf)
for more information of the method used.
"""
struct SpatialHashTable{Dim,VT<:AbstractVector,FT<:AbstractFloat} <: AbstractSpatialHashTable
    cellcount::VT
    particlemap::VT

    tablesize::Int64
    inv_cellsize::SVector{Dim,FT}

    pseudorandom_factors::SVector{Dim,Int64}

    caches::Vector{Set{Int64}}
end

"""
    dimension(ht::AbstractSpatialHashTable)

Returns the spatial dimension of a hash table.
"""
dimension(::SpatialHashTable{Dim}) where {Dim} = Dim
dimension(::BoundedHashTable{Dim}) where {Dim} = Dim

"""
    inttype(ht::AbstractSpatialHashTable)

Returns the integer type used for indexing the hash table. (Useful for GPU support.)
"""
inttype(ht::AbstractSpatialHashTable) = eltype(ht.cellcount)

"""
    BoundedHashTable(N::Integer, grid::Tuple, domainstart::SVector, domainend::SVector)

Primary constructor for a `BoundedHashTable` with `N` particles, a grid size of `grid` and 
a domain from `domainstart` to `domainend`. The grid is a tuple of size `Dim` which 
determines the number of cells along each dimension.

We refer to the other constructors for more convenient ways to construct a `BoundedHashTable`.
"""
function BoundedHashTable(N::Integer, grid::Tuple, domainstart::SVector, domainend::SVector)

    cellcount = Vector{typeof(N)}(undef, prod(grid) + 1)
    particlemap = Vector{typeof(N)}(undef, N)

    inv_cellsize = grid ./ (domainend - domainstart)
    strides = (oneunit(eltype(grid)), cumprod(grid[1:end-1])...)

    return BoundedHashTable(cellcount, particlemap, domainstart, domainend, inv_cellsize, strides, grid)
end


"""
    BoundedHashTable(N::Integer, grid, domainstart, domainend)

Same as the main constructor, but allowing the domain to be specified by `Vector{Float64}` 
or `NTuple{Dim,Float64}` instead of `StaticVectors`
"""
function BoundedHashTable(N::Integer, grid, domainstart, domainend)
    domainstart = SVector{length(grid),Float64}(domainstart)
    domainend = SVector{length(grid),Float64}(domainend)
    return BoundedHashTable(N, grid, domainstart, domainend)
end


"""
    BoundedHashTable(N::Integer, grid::Tuple, range::AbstractVector)

Constructs a `BoundedHashTable` with `N` particles, a grid size of `grid` and a domain
from `zero(range)` to `range`. See details of the main constructor for more information.
"""
BoundedHashTable(N::Int64, grid, range) = BoundedHashTable(N, grid, zero(range), range)

"""
    BoundedHashTable(N::Int64, cellsize::Number, domainstart::SVector, domainend::SVector)

Constructs a `BoundedHashTable` with `N` particles, a cell size of `cellsize` and a domain
from `domainstart` to `domainend`. The cell size is the same along each dimension.
"""
function BoundedHashTable(N::Int64, cellsize::Number, domainstart::SVector, domainend::SVector)
    grid = @. max(1, floor(Int64, (domainend - domainstart) / cellsize))
    return BoundedHashTable(N, tuple(grid...), domainstart, domainend)
end

"""
    BoundedHashTable(N::Int64, cellsize::Number, domainstart, domainend)

Same as previous constructor, but allowing the domain to be specified by `Vector{Float64}`
or `NTuple{Dim,Float64}` instead of `StaticVectors`. `cellsize` determines the cell size in
each dimension.
"""
function BoundedHashTable(N::Int64, cellsize::Number, domainstart, domainend)
    domainstart = SVector{length(domainstart),Float64}(domainstart)
    domainend = SVector{length(domainstart),Float64}(domainend)
    return BoundedHashTable(N, cellsize, domainstart, domainend)
end

"""
    BoundedHashTable(X::AbstractVector, args...)

Constructs a `BoundedHashTable` with `length(X)` particles and calls 
`updatetable!(ht, X)` to update the hash table.
"""
function BoundedHashTable(X::AbstractVector, args...)
    ht = BoundedHashTable(length(X), args...)
    updatetable!(ht, X)
    return ht
end

const default_factors = (92837111, 689287499, 283923481)

"""
    SpatialHashTable(N::Integer, cellsize::SVector, tablesize::Integer; cachesize=Threads.nthreads(), pseudorandomfactors=default_factors)

Primary constructor for a `SpatialHashTable` with `N` particles, a table size of `tablesize` and
a cell size of `cellsize`. The table size determines the number of cells in the hash table.
The cell size is a `SVector` of size `Dim` which determines the size of each cell along each dimension.

Additionally, the `cachesize` determines the number of thread-local caches used for the hash table.
Once can set `cachesize=1` for single threaded applications.

The `pseudorandomfactors` are used to generate a hash index for each particle. The default values
are taken from [Matthias Müller's paper](https://matthias-research.github.io/pages/publications/tetraederCollision.pdf).
"""
function SpatialHashTable(N::Int64, cellsize::SVector, tablesize::Int64; cachesize=Threads.nthreads(), pseudorandomfactors=default_factors)
    cellcount = Vector{Int64}(undef, tablesize + 2) # we need one extra for Julia type stability...
    particlemap = Vector{Int64}(undef, N)

    inv_cellsize = 1 ./ cellsize

    Dim = length(cellsize)
    pseudorandomfactors = SVector{Dim,Int64}(pseudorandomfactors[i] for i in eachindex(inv_cellsize))

    caches = [Set{Int64}() for i in 1:cachesize]

    return SpatialHashTable(cellcount, particlemap, tablesize, inv_cellsize, pseudorandomfactors, caches)
end

"""
    SpatialHashTable(N::Integer, cellsize::Number, tablesize::Integer; args...)

Same as the main constructor, but allowing the cell size to be specified by a tuple of size `Dim`.
"""
SpatialHashTable(N::Int64, cellsize, tablesize::Int64, args...) = SpatialHashTable(N, SVector{length(cellsize),Float64}(cellsize), tablesize, args...)

"""
    SpatialHashTable(N::Integer, cutoff::Number, Dim::Integer, tablesize::Integer; args...)

Same as the main constructor, but allowing the cell size to be specified by a number `cutoff` and
the spatial dimension `Dim`.
"""
function SpatialHashTable(N::Int64, cutoff::Number, Dim::Int64, tablesize::Int64, args...)
    return SpatialHashTable(N, fill(cutoff, Dim), tablesize, args...)
end

"""
    SpatialHashTable(X::AbstractVector, cutoff::Number, tablesize::Int64, args...)

Constructs a `SpatialHashTable` with `length(X)` particles and calls 
`updatetable!(ht, X)` to update the hash table. The `cutoff` determines the cell size along each dimension
and the `Dim` is inferred from the first particle in `X`.
(Alternatively one can provide a `SVector` of size `Dim` or a `NTuple{Dim,Float64}`.) 
"""
function SpatialHashTable(X::AbstractVector, cutoff::Number, tablesize::Int64, args...)
    Dim = length(X[1])
    ht = SpatialHashTable(length(X), cutoff, Dim, tablesize, args...)
    updatetable!(ht, X)
    return ht
end

"""
    SpatialHashTable(X::AbstractVector, cutoff::Number, tablesize::Int64, args...)

Constructs a `SpatialHashTable` with `length(X)` particles and calls 
`updatetable!(ht, X)` to update the hash table.
"""
function SpatialHashTable(X::AbstractVector, args...)
    ht = SpatialHashTable(length(X), args...)
    updatetable!(ht, X)
    return ht
end



"""
    updatetable!(ht::AbstractSpatialHashTable, X)

Updates the hash table `ht` with the positions `X` of the particles.

This method is linear in the number of particles and should generally not allocate.

The ideal datatype for `X` is `Vector{SVector{Dim,FT}}` where `Dim` is the spatial dimension
and `FT` is the floating point type.	
"""
function updatetable!(ht::AbstractSpatialHashTable, X)
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

"""
    resize!(ht::AbstractSpatialHashTable, n_positions)

Resizes the hash table `ht` to accomodate `n_positions` particles. This method is useful
if the number of particles changes over time. 

This call should usually be followed by a call to `updatetable!` to update the hash table.
Without `updatetable!` the hash table will be in an inconsistent state.
"""
function Base.resize!(ht::AbstractSpatialHashTable, n_positions)
    resize!(ht.particlemap, n_positions)
end


"""
    inside(ht::AbstractSpatialHashTable, pos)

Returns `true` if `pos` is inside the domain of the hash table `ht`.
"""
insidegrid(ht::BoundedHashTable, gridpos) = all(@. 1 <= gridpos <= ht.gridsize)
gridindices(ht::BoundedHashTable, pos) = ceil.(inttype(ht), (pos - ht.domainstart) .* ht.inv_cellsize)
hashindex(ht::BoundedHashTable, gridindices) = sum(@. (gridindices - 1) * ht.strides) + 1

insidegrid(::SpatialHashTable, gridpos) = true
gridindices(ht::SpatialHashTable, pos) = ceil.(inttype(ht), @. pos * ht.inv_cellsize)

function hashindex(ht::SpatialHashTable, gridindices)
    return oneunit(inttype(ht)) + mod(abs(reduce(⊻, gridindices .* ht.pseudorandom_factors)), ht.tablesize)
end

"""
    hashposition(ht::AbstractSpatialHashTable, pos)

Returns the hash position of `pos` in the hash table `ht`.
"""
hashposition(ht::AbstractSpatialHashTable, pos) = hashindex(ht, gridindices(ht, pos))


"""
    iterate_box(ht::AbstractSpatialHashTable, boxhash)

Returns an iterator over the particles in the box with hash `boxhash` in the hash table `ht`.
"""
function iterate_box(ht::AbstractSpatialHashTable, boxhash)
    box_start = ht.cellcount[boxhash] + oneunit(inttype(ht))
    box_end = ht.cellcount[boxhash+1]
    return (ht.particlemap[k] for k in box_start:box_end)
end

"""
    neighbouring_boxes(ht::AbstractSpatialHashTable, gridpos, r)

Returns an iterator over the neighbouring boxes of the box with grid position `gridpos` in the hash table `ht`.
`r` determines the radius of the neighbourhood.

The `gridpos` is a tuple of size `Dim` which determines the position of a box in lattice 
with basis vectors `ht.inv_cellsize`.
"""
function neighbouring_boxes(ht::AbstractSpatialHashTable, gridpos, r)
    IT = inttype(ht)
    Dim = IT(dimension(ht))
    widths = @. ceil(IT, r * ht.inv_cellsize)
    int_offsets = CartesianIndices(ntuple(i -> -widths[i]:widths[i], Dim))
    offsets = (gridpos .+ Tuple(i) for i in int_offsets)
    return (hashindex(ht, offset) for offset in offsets if insidegrid(ht, offset))
end

"""
    neighbours(ht::AbstractSpatialHashTable, pos, r)

Returns an iterator over the particles in the neighbourhood of `pos` in the hash table `ht`.
`r` determines the radius of the neighbourhood.

This is the main method of this package and is used to find the neighbours of a particle.
"""
function neighbours(ht::AbstractSpatialHashTable, pos, r)
    gridpos = gridindices(ht, pos)
    return (k for boxhash in neighbouring_boxes(ht, gridpos, r) for k in iterate_box(ht, boxhash))
end


# The following code deals with hash index collisions which could result 
# in the same index being returned multiple times.
thread_cache(ht::SpatialHashTable) = ht.caches[Threads.threadid()]

function iterate_box_if(ht, boxhash, seen)
    if boxhash in seen
        return iterate_box(ht, ht.tablesize + 1)  # this box is empty (see constructor where we add +2)
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



"""
    Adapt.adapt_structure(to, ht::AbstractSpatialHashTable)

Adapts the hash table `ht` to the type `to`. This is useful for GPU support.

It automatically converts types to `Float64` and `Int32`!
"""
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


export SpatialHashTable, BoundedHashTable, AbstractSpatialHashTable
export updatetable!, resize!, neighbours, iterate_box, dimension, inside, hashposition
end
