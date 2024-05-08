using Revise 

using CellListMap
using SpatialHashTables
using ProtoStructs
using CUDA, KernelAbstractions
using LinearAlgebra, StaticArrays
using ThreadsX




@kernel function compute_cell_indices_kernel!(hg, X)
    tid = @index(Global)
    hg.pointcellidx[tid] = pos2hash(hg, X[tid])
    hg.pointidx[tid] = tid 
end

function compute_cell_indices!(hg, X, nthreads = hg.nthreads)
    compute_cell_indices_kernel!(hg.backend, unval(nthreads))(hg, X, ndrange = length(X))
end

@kernel function compute_cell_offsets_kernel!(hg)
    tid = @index(Global)

    c = hg.pointcellidx[tid]
    if tid == 1
        hg.cellstarts[c] = 1
    else
        p = hg.pointcellidx[tid-1]
        if c != p
            hg.cellstarts[c] = tid 
            hg.cellends[p] = tid-1
        end
    end

    if tid == length(hg.pointidx)
        hg.cellends[c] = length(hg.pointidx)
    end
end

function compute_cell_offsets!(hg, nthreads = hg.nthreads)
    compute_cell_offsets_kernel!(hg.backend, unval(nthreads))(hg, ndrange = length(hg.pointcellidx))
end


function paired_sort!(ix, a, backend, nthreads)
    if backend isa CUDABackend
        CUDA.sortperm!(ix, a)
        CUDA.permute!(ix, a)
    elseif backend isa CPU && unval(nthreads) > 1
        sortperm!(ix, a, alg = ThreadsX.QuickSort)
        permute!(ix, a)
    else
        sortperm!(ix, a)
        permute!(ix, a)
    end
end

function updatecells!(hg, X, nthreads = hg.nthreads)    
    compute_cell_indices!(hg, X, nthreads)
    paired_sort!(hg.pointidx, hg.pointcellidx, hg.backend, nthreads)
    compute_cell_offsets!(hg, nthreads)
    KernelAbstractions.synchronize(hg.backend)
end



cu 
SVec3 = SVector{3,Float32}

X_cpu, _ = CellListMap.xatomic(10^6)

# cell list map 
lim = limits(X_cpu)
box = Box(lim, 12.0)
cl = CellList(X_cpu, box)
r = box.cutoff

#X_cpu = rand(SVec3, 10^6)
X = cu(convert.(SVec3,X_cpu))


hg = HashGrid(CuArray{Int32}, X, Float32.(box.cell_size), box.nc)
hg_cpu = HashGrid(Vector{Int64}, X_cpu, box.cell_size, box.nc)

@time updatecells!(hg, X)
@time updatecells!(hg_cpu, X_cpu)

# system = InPlaceNeighborList(x=X_cpu, cutoff = box.cutoff, unitcell = lim.limits)
# @time update!(system, X_cpu)

# @profview updatecells!(hg_cpu, X_cpu)

function cellcontent(hg, cellind)

end

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