using Revise 

using CellListMap
using SpatialHashTables

X, _ = CellListMap.xatomic(10^5)
lim = limits(X)

X_cu = cu(X)

# cell list map 
box = Box(lim, 12.0)
cl = CellList(X, box)
r = box.cutoff

bht = BoundedHashTable(X, r, lim.limits .+ 12)

using ProtoStructs

@proto struct BoxQuery{VT} 
    boxposition::VT
    boxranges::RT 
end

BoxQuery( [1,2])


function create_query(ht, r, pos)
    starts = @. round(Int, (pos - r) * ht.inv_cellsize)
    ends   = @. min(round(Int, (pos + r) * ht.inv_cellsize), starts + ht.gridsize - 1)

    ind = starts 
    cells = hashindex(ht, ind)
    cell_index = ht.cell_starts[cell]
    cell_end = ht.cell_ends[cell]    

    return (starts, ends)
end

create_query(bht, 12.0, X[1])

neighbours = CellQuery...

for i in eachindex(X) 
    Xi = X[i]
    for j in cq(Xi)

    end
end

function hash_grid_index(ht, pos)
    origin = 1<<20
    return dot(max(0, pos + origin) % ht.grid_size, ht.strides)
end


struct HashGrid 
    cell_width 
    cell_width_inv 

    point_cells 
    point_ids 

    cell_starts 
    cell_ends 

    grid_size 
    strides 

    num_points 
    max_points 

    context
end

function HashGrid(cell_width, num_points, max_points = 2 * num_points, context = nothing)
    cell_width_inv = 1 ./ cell_width


end

function compute_cell_indices_kernel(grid, points, i)
    grid.point_cells[i] = hash_grid_index(ht, points[i])
    grid.point_ids[i] = i
end

using CUDA

CUDA.fill(0.0f0, 10)

function compute_cell_indices(grid, points::CuArray)
    # ...
end

function compute_cell_indices(grid, points)
    for i in eachindex(points)
        compute_cell_indices_kernel(grid, points, i)
    end
end