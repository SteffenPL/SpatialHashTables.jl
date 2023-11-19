
"""
    AbstractSpatialHashTable

Abstract type for spatial hash tables.

Expected to implement the following functions:
    dimension(::AbstractSpacialHashTable)
    inttype(ht::AbstractSpatialHashTable)
    insidegrid(ht::BoundedHashTable, gridpos)
    gridindices(ht::BoundedHashTable, pos)
    hashindex(ht::BoundedHashTable, gridindices)

Moreover, we assume that the hash table has the following fields:
    ht.cellcount
    ht.particlemap
"""
abstract type AbstractSpatialHashTable end

"""
    inttype(ht::AbstractSpatialHashTable)

Returns the integer type used for indexing the hash table. (Useful for GPU support.)
"""
inttype(ht::AbstractSpatialHashTable) = eltype(ht.cellcount)

"""
    hashposition(ht::AbstractSpatialHashTable, pos)

Returns the hash position of `pos` in the hash table `ht`.
"""
hashposition(ht::AbstractSpatialHashTable, pos) = hashindex(ht, gridindices(ht, pos))
