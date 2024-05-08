struct HashGrid{Dim, FltT<:Real, IntVecT, IntT, BackendT, NThreads} 
    
    cellwidth::SVector{Dim,FltT}
    cellwidthinv::SVector{Dim,FltT}

    gridsize::NTuple{Dim,IntT} 
    
    origin::SVector{Dim,FltT}

    cellidx::IntVecT 
    pointidx::IntVecT

    cellstarts::IntVecT
    cellends::IntVecT 

    backend::BackendT
    nthreads::NThreads
end

# basic information

dimension(hg::HashGrid) = length(hg.cellwidth)
floattype(hg::HashGrid) = eltype(hg.cellwidth)
inttype(hg::HashGrid) = eltype(hg.gridsize)
linearindices(hg::HashGrid) = LinearIndices(hg.gridsize)
cartesianindices(hg::HashGrid) = CartesianIndices(hg.gridsize)

# access cells 

cell(hg::HashGrid, i) = (hg.pointidx[k] for k in hg.cellstarts[i]:hg.cellends[i])
cellpos(hg::HashGrid, i) = hg.origin + hg.cellwidth .* cartesianindices(hg)[i]

# constructors

function HashGrid{IndexVecT}(radius, gridsize, origin, npts, nthreads, backend) where {IndexVecT}
    Dim = length(gridsize)
    FltT = eltype(origin)

    cellwidth = SVector(ntuple(i -> get(radius, i, radius[end]), Dim))
    cellwidthinv = one(FltT) ./ cellwidth

    cellidx  = IndexVecT(undef, npts)
    pointidx = IndexVecT(undef, npts)

    ncells = prod(gridsize)
    cellstarts = IndexVecT(undef, ncells)
    cellends   = IndexVecT(undef, ncells)

    return HashGrid(cellwidth, cellwidthinv, gridsize, origin, cellidx, pointidx, cellstarts, cellends, backend, Val(nthreads))
end

function HashGrid{IndexVecType}(pts, radius, gridsize;
                    origin = zero(eltype(pts)), 
                    nthreads = eltype(IndexVecType) == Int32 ? 256 : Threads.nthreads()) where {IndexVecType}
    
    npts = length(pts)
    backend = get_backend(pts)
    return HashGrid{IndexVecType}(radius, gridsize, origin, npts, nthreads, backend)
end

HashGrid(pts::Vector{SVector{Dim,Float64}}, args...; kwargs...) where {Dim} = HashGrid{Vector{Int64}}(pts, args...; kwargs...)


# index functions
# - pos  (position, vector)
# - grid (gridindex, cartesian) 
# - hash (cellindex, linear)

@inline function pos2grid(grid, pos)
    IntT = inttype(grid)
    return @. mod1(ceil(IntT, (pos - grid.origin) * grid.cellwidthinv), grid.gridsize)
end

@inline grid2hash(grid, ind) = linearindices(grid)[ind...]
@inline pos2hash(grid, pos) = grid2hash(grid, pos2grid(grid, pos))


# update functions

@kernel function compute_cell_hashes_kernel!(hg, pts)
    tid = @index(Global)
    hg.cellidx[tid] = pos2hash(hg, pts[tid])
    hg.pointidx[tid] = tid 
end

function compute_cell_hashes!(hg, pts, nthreads = hg.nthreads)
    compute_cell_hashes_kernel!(hg.backend, unval(nthreads))(hg, pts, ndrange = length(pts))
end

@kernel function compute_cell_offsets_kernel!(hg)
    tid = @index(Global)

    c = hg.cellidx[tid]
    if tid == 1
        hg.cellstarts[c] = 1
    else
        p = hg.cellidx[tid-1]
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
    compute_cell_offsets_kernel!(hg.backend, unval(nthreads))(hg, ndrange = length(hg.cellidx))
end

function paired_sort!(ix, a, modul, alg)
    if isnothing(alg) && (!isnothing(modul) || a isa Vector)
        getproperty(modul, :sortperm!)(ix, a)
        getproperty(modul, :permute!)(ix, a)
    elseif !isnothing(alg)
        getproperty(modul, :sortperm!)(ix, a; alg)
        getproperty(modul, :permute!)(ix, a)
    else 
        error("""Please select a proper module and sorting algorithm for the vector type $(typeof(a))
        For example via `updatecells!(grid, pts; module = CUDA)`.
        """)
    end
end

function updatecells!(hg, pts; modul = Base, alg = nothing, nthreads = hg.nthreads)    

    resize!(hg.cellidx, length(pts))
    resize!(hg.pointidx, length(pts))

    compute_cell_hashes!(hg, pts, nthreads)
    paired_sort!(hg.pointidx, hg.cellidx, modul, alg)
    compute_cell_offsets!(hg, nthreads)

    # after this operation, pointidx[i] is a point index and cellidx[i] is the cell index containing the point 
    # cellstarts and cellends defines the range of a cell inside 'pointidx' 

    KernelAbstractions.synchronize(hg.backend)
end



# iteration over neighbours 

struct HashGridQuery{HG <: HashGrid, CIndsT <: CartesianIndices}
    grid::HG
    cellindices::CIndsT
end

function HashGridQuery(hg::HashGrid, pos, r)
    starts =      pos2index(hg, pos .- r)
    ends   = min.(pos2index(hg, pos .+ r), @. starts + hg.gridsize - 1)
    cellindices = CartesianIndices( ntuple( i -> starts[i]:ends[i], Dim(hg)))

    return HashGridQuery(cellindices, hg)
end

neighbours(hg, pos, r) = HashGridQuery(hg, pos, r)


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