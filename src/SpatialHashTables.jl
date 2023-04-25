module SpatialHashTables

    # Write your package code here.
    using StaticArrays

    struct SpatialHashTable{N} 
        cellcount::Vector{Int64}
        particlemap::Vector{Int64}
        domain::NamedTuple{(:min,:max), NTuple{2, SVector{N, Float64}}}
        grid::NTuple{N, Int64}
        spacing::SVector{N, Float64}
        linear_indices::LinearIndices{N, NTuple{N, Base.OneTo{Int64}}}
        cartesian_indices::CartesianIndices{N, NTuple{N, Base.OneTo{Int64}}}
    end 
    
    function boxindex(ht::SpatialHashTable, pos)
        return ht.linear_indices[ceil.(Int64, (pos - ht.domain.min) ./ ht.spacing )...]
    end
    
    function SpatialHashTable(domain, grid, n_positions)
        cellcount = Vector{Int64}(undef, prod(grid)+1)
        particlemap = Vector{Int64}(undef, n_positions)
        spacing = (domain.max - domain.min) ./ grid
        linear_indices = LinearIndices(grid)
        cartesian_indices = CartesianIndices(grid)
        return SpatialHashTable(cellcount, particlemap, domain, grid, spacing, linear_indices, cartesian_indices)
    end 
    
    function SpatialHashTable(domain, grid, X::AbstractVector)
        ht = SpatialHashTable(domain, grid, length(X))
        updateboxes!(ht, X)
        return ht
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
    
    function iterate_box(ht::SpatialHashTable, box_index) 
        box_start = ht.cellcount[box_index] + 1
        box_end = ht.cellcount[box_index+1]
        return ( ht.particlemap[k] for k in box_start:box_end )
    end 
    
    function neighbouring_boxes(ht::SpatialHashTable{N}, box_index, r) where N
        width = ceil(Int64, r / minimum(ht.spacing) )
        offsets = CartesianIndices( (-width:width, -width:width) )
        box = ht.cartesian_indices[box_index]
        return ( ht.linear_indices[box + offset] for offset in offsets if checkbounds(Bool, ht.linear_indices, box + offset) )
    end
    
    function neighbours(ht::SpatialHashTable, pos, r)
        box = boxindex(ht, pos)
        return ( k for bj in neighbouring_boxes(ht, box, r) for k in iterate_box(ht, bj) )
    end
    

    export SpatialHashTable, updateboxes!, resize!, neighbours, boxindex, neighbouring_boxes, iterate_box

end
