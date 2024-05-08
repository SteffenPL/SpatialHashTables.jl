using Revise 

using CellListMap
using SpatialHashTables
using ProtoStructs
using CUDA, KernelAbstractions
using LinearAlgebra, StaticArrays
using ThreadsX

unval(::Val{x}) where {x} = x

struct HashGrid{Dim, NThreads, IntT <: Integer, FltT <: Real, IntVecT <: AbstractVector, BT} 
    cellwidth::NTuple{Dim,FltT}
    cellwidthinv::NTuple{Dim,FltT}

    pointcellidx::IntVecT 
    pointidx::IntVecT

    cellstarts::IntVecT
    cellends::IntVecT 
    numcells::IntT

    gridsize::NTuple{Dim,IntT} 
    strides::NTuple{Dim,IntT}

    backend::BT
    nthreads::NThreads
end

Dim(hg::HashGrid) = length(hg.cellwidth)
FloatType(hg::HashGrid) = eltype(hg.cellwidth)
IntType(hg::HashGrid) = eltype(hg.gridsize)

function HashGrid(IndexVecType::Type, points, radius, gridsize, 
                    nthreads = eltype(points) == Float32 ? 1024 : Threads.nthreads())
    IntT = eltype(IndexVecType)
    gridsize = convert.(IntT, gridsize)
    Dim = length(gridsize)

    cellwidth = ntuple(i -> get(radius, i, radius[end]), Dim)
    cellwidthinv = 1 ./ cellwidth

    pointcellidx = IndexVecType(undef, length(points))
    pointidx = IndexVecType(undef, length(points))

    strides = convert.(IntT, cumprod((1,gridsize...)[1:end-1]))
    numcells = convert(IntT, prod(gridsize))

    cellstarts = IndexVecType(undef, numcells)
    cellends = IndexVecType(undef, numcells)

    return HashGrid(cellwidth, cellwidthinv, pointcellidx, pointidx, cellstarts, cellends, numcells, Tuple(gridsize), Tuple(strides), get_backend(points), Val(nthreads))
end


using Adapt
function Adapt.adapt_structure(to, hg::HashGrid) 
    HashGrid(
        hg.cellwidth,
        hg.cellwidthinv,
        Adapt.adapt_structure(to, hg.pointcellidx),
        Adapt.adapt_structure(to, hg.pointidx),
        Adapt.adapt_structure(to, hg.cellstarts),
        Adapt.adapt_structure(to, hg.cellends), 
        hg.numcells,
        hg.gridsize,
        hg.strides,
        hg.backend,
        hg.nthreads)
end


@inline function index2hash(hg, ind)
    IntT = IntType(hg)
    return sum(@. (mod1(ind, hg.gridsize) - one(IntT)) * hg.strides) + one(IntT)
end

@inline function pos2hash(hg, pos)
    IntT = IntType(hg)
    return index2hash(hg, @.(ceil(IntT,pos * hg.cellwidthinv)))
end


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



struct HashGridQuery{IT <: Integer, PT, RT, HG}
    boxposition::PT
    boxranges::RT
    centerindex::IT  # cell index of the center position 
    cellindex::IT    # 
    cellend::IT
    grid::HG
end


function createquery(hg, pos, r)
    IntT = IntType(hg)
    starts = @. round(IntT, (pos - r) * hg.inv_cellsize)
    ends   = @. min(round(IntT, (pos + r) * ht.inv_cellsize), starts + ht.gridsize - 1)

    firstcell = starts 
    cell = hashindex(ht, ind)
    cellpointidx = ht.cell_starts[cell]
    cell_end = ht.cell_ends[cell]    

    return (starts, ends)
end

function next_in

@kernel function compute_energy(forces, X, hg)
    tid = @index(Global) 

    query = create_query(hg, X[tid], 12.0)

    for i in query 
        xij = X[tid] - X[i]
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