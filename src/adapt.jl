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
