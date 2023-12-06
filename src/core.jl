
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
    neighbour_indices = CartesianIndices(ntuple(i -> -widths[i]:widths[i], Dim))
    neighbour_reps = (gridpos .+ Tuple(i) for i in neighbour_indices)
    return (hashindex(ht, rep) for rep in neighbour_reps if insidegrid(ht, rep))
end

"""
    neighbours(ht::AbstractSpatialHashTable, pos, r)

Returns an iterator over the particles in the neighbourhood of `pos` in the hash table `ht`.
`r` determines the radius of the neighbourhood.

This is the main method of this package and is used to find the neighbours of a particle.
"""
function neighbours(ht::AbstractSpatialHashTable, pos, r)
    gridpos = gridindices(ht, pos)
    return (k   for boxhash in neighbouring_boxes(ht, gridpos, r) 
                for k in iterate_box(ht, boxhash))
end

@inline function wrap_index(ht::BoundedHashTable, gridpos)
    rep = @. mod(gridpos - 1, ht.gridsize) + 1
    offset = @. ceil(Int64, (rep - gridpos) / ht.gridsize) * ht.domainsize
    return (rep = rep, offset = offset)
end

function periodic_neighbouring_boxes(ht::BoundedHashTable, gridpos, r)
    IT = inttype(ht)
    Dim = IT(dimension(ht))
    widths = @. ceil(IT, r * ht.inv_cellsize)
    neighbour_indices = CartesianIndices(ntuple(i -> -widths[i]:widths[i], Dim))
    neighbour_reps = (wrap_index(ht, gridpos .+ Tuple(i)) for i in neighbour_indices)
    return ( (hashindex(ht, rep), offset) for (rep, offset) in neighbour_reps)
end

function periodic_neighbours(ht::BoundedHashTable, pos, r)
    gridpos = gridindices(ht, pos)
    return ((k, offset) for (boxhash, offset) in periodic_neighbouring_boxes(ht, gridpos, r) 
                for k in iterate_box(ht, boxhash))
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
