using Revise 

using CellListMap
using SpatialHashTables
using ProtoStructs
using CUDA, KernelAbstractions
using LinearAlgebra, StaticArrays
using ThreadsX


SVec3 = SVector{3,Float32}
X_cpu, _ = CellListMap.xatomic(10^6)
X = cu(convert.(SVec3,X_cpu))

# cell list map 
lim = limits(X_cpu)
box = Box(lim, 12.0)
cl = CellList(X_cpu, box)
r = box.cutoff

gridsize = Tuple(box.nc)
cellwidth = box.cell_size

hg = HashGrid{CuVector{Int32}}(X, Float32.(cellwidth), Int32.(gridsize); nthreads = 1024)
hg_cpu = HashGrid(X_cpu, cellwidth, gridsize)

@time updatecells!(hg, X; modul = CUDA)
@time updatecells!(hg_cpu, X_cpu)

# system = InPlaceNeighborList(x=X_cpu, cutoff = box.cutoff, unitcell = lim.limits)
# @time update!(system, X_cpu)

# @profview updatecells!(hg_cpu, X_cpu)

struct HashGridQuery{CIndsT <: CartesianIndices, HG <: HashGrid}
    cellindices::CIndsT
    grid::HG
end

function neighbours(hg, pos, r)
    starts =      pos2index(hg, pos .- r)
    ends   = min.(pos2index(hg, pos .+ r), @. starts + hg.gridsize - 1)
    cellindices = CartesianIndices( ntuple( i -> starts[i]:ends[i], Dim(hg)))

    return HashGridQuery(cellindices, hg)
end

function Base.iterate(query::HashGridQuery)
    cellind = first(query.cellindices)
    linearidx = LinearIndices(query.grid.gridsize)[cellind]
    i       = query.grid.cellstarts[linearidx]
    cellend = query.grid.cellends[linearidx]

    initstate = (cellind, i-1, cellend)

    return iterate(query, initstate)
end

function Base.iterate(query::HashGridQuery, state)
    (cellind, i, cellend) = state

    while true
        if i <= cellend
            k = query.grid.pointidx[i]
            return (k, (cellind, i+1, cellend))
        else
            next = iterate(query.cellindices, cellind)

            if isnothing(next)
                return nothing 
            end

            cellind = next[1]
            linearidx = LinearIndices(query.grid.gridsize)[cellind]
            i       = query.grid.cellstarts[linearidx]
            cellend = query.grid.cellends[linearidx]
        end
    end
end




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