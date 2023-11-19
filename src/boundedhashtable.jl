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
    BoundedHashTable(N::Int64, cutoff::Number, domainstart::SVector, domainend::SVector)

Constructs a `BoundedHashTable` with `N` particles. The cell size is close to `cutoff` (or slightly larger) 
in each dimension. The domain ranges from `domainstart` to `domainend`.
"""
function BoundedHashTable(N::Int64, cutoff::Number, domainstart::SVector, domainend::SVector)
    grid = @. max(1, floor(Int64, (domainend - domainstart) / cutoff))
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

dimension(::BoundedHashTable{Dim}) where {Dim} = Dim

"""
    inside(ht::AbstractSpatialHashTable, pos)

Returns `true` if `pos` is inside the domain of the hash table `ht`.
"""
insidegrid(ht::BoundedHashTable, gridpos) = all(@. 1 <= gridpos <= ht.gridsize)
gridindices(ht::BoundedHashTable, pos) = ceil.(inttype(ht), (pos - ht.domainstart) .* ht.inv_cellsize)
hashindex(ht::BoundedHashTable, gridindices) = sum(@. (gridindices - 1) * ht.strides) + 1
