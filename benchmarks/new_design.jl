using Revise 
using LinearAlgebra, StaticArrays

using CellListMap
using SpatialHashTables
using CUDA, ThreadsX

using SpatialHashTables: cell, cellpos

SVec3 = SVector{3,Float32}
X_cpu, _ = CellListMap.xatomic(10^4)
X = cu(convert.(SVec3,X_cpu))

# cell list map 
lim = limits(X_cpu)
box = Box(lim, 50.0)
cl = CellList(X_cpu, box)
r = box.cutoff

gridsize = Tuple(box.nc)
cellwidth = box.cell_size

cellwidth = 25.0
gridsize = (2, 2, 2)

hg = HashGrid{CuVector{Int32}}(X, Float32.(cellwidth), Int32.(gridsize); nthreads = 1024)
hg_cpu = HashGrid(X_cpu, cellwidth, gridsize)

@time updatecells!(hg, X; modul = CUDA)
@time updatecells!(hg_cpu, X_cpu)

@kernel function compute_energy(forces, X, hg)
    tid = @index(Global) 

    Xi = X[tid]

    for j in neighbours(hg, X[tid], 12.0) 
        xij = Xi - X[j]
        forces[tid] = xij / norm(xij)
    end

end 

forces = similar(X)
compute_energy(hg.backend, unval(hg.nthreads))(forces, X, hg, ndrange = length(X))



create_query(bht, 12.0, X[1])

neighbours = CellQuery...

for i in eachindex(X) 
    Xi = X[i]
    for j in cq(Xi)

    end
end



# struct HashGrid 
#     cell_width 
#     cell_width_inv 

#     point_cells 
#     point_ids 

#     cell_starts 
#     cell_ends 

#     grid_size 
#     strides 

#     num_points 
#     max_points 

#     context
# end

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