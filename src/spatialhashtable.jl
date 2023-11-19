const default_factors = (92837111, 689287499, 283923481)


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
    SpatialHashTable(N::Integer, cellsize, tablesize::Integer; args...)

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
    dimension(ht::AbstractSpatialHashTable)

Returns the spatial dimension of a hash table.
"""
dimension(::SpatialHashTable{Dim}) where {Dim} = Dim


insidegrid(::SpatialHashTable, gridpos) = true
gridindices(ht::SpatialHashTable, pos) = ceil.(inttype(ht), @. pos * ht.inv_cellsize)
function hashindex(ht::SpatialHashTable, gridindices)
    return oneunit(inttype(ht)) + mod(abs(reduce(⊻, gridindices .* ht.pseudorandom_factors)), ht.tablesize)
end
