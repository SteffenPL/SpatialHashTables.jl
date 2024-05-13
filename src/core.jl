struct HashGrid{Dim, FltT<:Real, IntVecT, IntT, LIndT, BackendT, NThreads} 
    
    cellwidth::SVector{Dim,FltT}
    cellwidthinv::SVector{Dim,FltT}

    gridsize::NTuple{Dim,IntT} 
    
    origin::SVector{Dim,FltT}

    cellidx::IntVecT 
    pointidx::IntVecT

    cellstarts::IntVecT
    cellends::IntVecT 

    lininds::LIndT

    backend::BackendT
    nthreads::NThreads
end

# basic information

dimension(grid::HashGrid) = length(grid.gridsize)
inttype(grid::HashGrid) = eltype(grid.gridsize)
linearindices(grid::HashGrid) = grid.lininds
cartesianindices(grid::HashGrid) = CartesianIndices(grid.gridsize)
floattype(grid::HashGrid) = eltype(grid.cellwidth)
numthreads(grid::HashGrid) = unval(grid.nthreads)

# access cells 

Base.size(grid::HashGrid) = grid.gridsize
Base.length(grid::HashGrid) = prod(size(grid))


function celldomain(grid::HashGrid, i)
    gridind = Tuple(cartesianindices(grid)[i])
    return (grid.origin + grid.cellwidth .* (gridind .- 1), grid.origin + grid.cellwidth .* gridind)
end 

cellsize(grid::HashGrid, i) = grid.cellwidth

Base.getindex(grid::HashGrid, k::Integer) = view(grid.pointidx, grid.cellstarts[k]:grid.cellends[k])

function Base.getindex(grid::HashGrid, i...) 
    k = linearindices(grid)[i...]
    return view(grid.pointidx, grid.cellstarts[k]:grid.cellends[k])
end

function domainsize(grid::HashGrid)
    size(grid) .* grid.cellwidth
end

function domain(grid::HashGrid)
    d = domainsize(grid)
    return (grid.origin, grid.origin + d)
end

# constructors

function HashGrid{IndexVecT}(radius, gridsize, origin, npts, nthreads, backend = KernelAbstractions.CPU()) where {IndexVecT}
    Dim = length(gridsize)
    FltT = eltype(origin)

    cellwidth = SVector(ntuple(i -> @compat(get(radius, i, radius[end])), Dim))
    cellwidthinv = one(FltT) ./ cellwidth

    cellidx  = IndexVecT(undef, npts)
    pointidx = IndexVecT(undef, npts)

    ncells = prod(gridsize)
    cellstarts = IndexVecT(undef, ncells)
    cellends   = IndexVecT(undef, ncells)

    lininds = LinearIndices(gridsize)

    return HashGrid(cellwidth, cellwidthinv, gridsize, origin, cellidx, pointidx, cellstarts, cellends, lininds, backend, Val(nthreads))
end

function HashGrid{IndexVecType}(pts, cellwidth, gridsize;
                    origin = zero(eltype(pts)), 
                    nthreads = eltype(IndexVecType) == Int32 ? 1024 : max(1,div(Threads.nthreads(),2))) where {IndexVecType}
    
    npts = length(pts)
    backend = get_backend(pts)
    grid = HashGrid{IndexVecType}(cellwidth, gridsize, origin, npts, nthreads, backend)
    updatecells!(grid, pts)
    return grid
end

HashGrid(pts::Vector{SVector{Dim,Float64}}, args...; kwargs...) where {Dim} = HashGrid{Vector{Int64}}(pts, args...; kwargs...)

function HashGrid{IndexVecType}(pts, domainstart, domainend, gridsize::NTuple; kwargs...) where {IndexVecType}
    cellwidth = (domainend .- domainstart) ./ gridsize
    return HashGrid{IndexVecType}(pts, cellwidth, gridsize; kwargs..., origin = domainstart)
end

function HashGrid{IndexVecType}(pts, domainstart, domainend, r::Real; kwargs...) where {IndexVecType}
    gridsize = Tuple(@. floor(eltype(IndexVecType), (domainend - domainstart) ./ r))
    cellwidth = (domainend .- domainstart) ./ gridsize
    return HashGrid{IndexVecType}(pts, cellwidth, gridsize; kwargs..., origin = domainstart)
end

# index functions
# - pos  (position, vector)
# - grid (gridindex, cartesian) 
# - hash (cellindex, linear)

@inline function pos2grid(grid, pos)
    IntT = inttype(grid)
    ind = @. mod1(ceil(IntT, (pos - grid.origin) * grid.cellwidthinv), grid.gridsize)
    return CartesianIndex(Tuple(ind))
end

@inline grid2hash(grid, ind) = linearindices(grid)[ind]
@inline pos2hash(grid, pos) = grid2hash(grid, pos2grid(grid, pos))

function pos2grid_unbound(grid, pos)
    IntT = inttype(grid)
    ind = @. ceil(IntT, (pos - grid.origin) * grid.cellwidthinv)
    return CartesianIndex(Tuple(ind))
end

# update functions
@kernel function compute_cell_hashes_kernel!(grid, pts)
    tid = @index(Global)
    grid.cellidx[tid] = pos2hash(grid, pts[tid])
    grid.pointidx[tid] = tid 
end

function compute_cell_hashes!(grid, pts, nthreads = grid.nthreads)
    compute_cell_hashes_kernel!(grid.backend, unval(nthreads))(grid, pts, ndrange = length(pts))
end

@kernel function compute_cell_offsets_kernel!(grid)
    tid = @index(Global)

    c = grid.cellidx[tid]
    if tid == 1
        grid.cellstarts[c] = 1
    else
        p = grid.cellidx[tid-1]
        if c != p
            grid.cellstarts[c] = tid 
            grid.cellends[p] = tid-1
        end
    end

    if tid == length(grid.pointidx)
        grid.cellends[c] = length(grid.pointidx)
    end
end

function compute_cell_offsets!(grid, nthreads = grid.nthreads)
    compute_cell_offsets_kernel!(grid.backend, unval(nthreads))(grid, ndrange = length(grid.cellidx))
end

function paired_sort!(ix, a, modul, alg)
    if isnothing(alg) && (!isnothing(modul) || a isa Vector)
        getproperty(modul, :sortperm!)(ix, a)
        getproperty(modul, :permute!)(a, ix)
    elseif !isnothing(alg)
        getproperty(modul, :sortperm!)(ix, a; alg)
        getproperty(modul, :permute!)(a, ix)
    else 
        error("""Please select a proper module and sorting algorithm for the vector type $(typeof(a))
        For example via `updatecells!(grid, pts; module = CUDA)`.
        """)
    end
end

function updatecells!(grid, pts; modul = Base, alg = nothing, nthreads = grid.nthreads)    

    resize!(grid.cellidx, length(pts))
    resize!(grid.pointidx, length(pts))

    fill!(grid.cellstarts, one(inttype(grid)))
    fill!(grid.cellends,  zero(inttype(grid)))

    compute_cell_hashes!(grid, pts, nthreads)
    paired_sort!(grid.pointidx, grid.cellidx, modul, alg)
    compute_cell_offsets!(grid, nthreads)

    # after this operation, pointidx[i] is a point index and cellidx[i] is the cell index containing the point 
    # cellstarts and cellends defines the range of a cell inside 'pointidx' 

    KernelAbstractions.synchronize(grid.backend)
end



# iteration over neighbours 

struct HashGridQuery{HG <: HashGrid, CIndsT <: CartesianIndices}
    grid::HG
    cellindices::CIndsT
end
Base.IteratorSize(::HashGridQuery) = Base.SizeUnknown()
Base.eltype(query::HashGridQuery) = eltype(query.grid.pointidx) 


function HashGridQuery(grid::HashGrid, pos, r)
    starts =     pos2grid_unbound(grid, pos .- r)
    ends   = min(pos2grid_unbound(grid, pos .+ r), CartesianIndex(Tuple(starts) .+ grid.gridsize .- 1))
    cellindices = starts:ends
    
    return HashGridQuery(grid, cellindices)
end

function warplinear(grid, ind) 
    return linearindices(grid)[CartesianIndex(mod1.(Tuple(ind), grid.gridsize))]
end 

function Base.iterate(query::HashGridQuery)
    cellind = first(query.cellindices)
    linearidx = warplinear(query.grid, cellind)
    i       = query.grid.cellstarts[linearidx]
    cellend = query.grid.cellends[linearidx]

    initstate = (cellind, i, cellend)

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
            linearidx = warplinear(query.grid, cellind)
            i       = query.grid.cellstarts[linearidx]
            cellend = query.grid.cellends[linearidx]
        end
    end 
end