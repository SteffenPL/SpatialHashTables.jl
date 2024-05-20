struct HashGrid{Dim, FltT<:Real, IntVecT, IntT, BackendT, NThreads} 
    
    cellwidth::SVector{Dim,FltT}
    cellwidthinv::SVector{Dim,FltT}

    gridsize::NTuple{Dim,IntT} 
    
    origin::SVector{Dim,FltT}

    cellidx::IntVecT 
    pointidx::IntVecT

    cellstarts::IntVecT
    cellends::IntVecT 

    strides::NTuple{Dim,IntT}

    backend::BackendT
    nthreads::NThreads
end

# basic information

dimension(grid::HashGrid) = length(grid.gridsize)
inttype(grid::HashGrid) = eltype(grid.gridsize)
oneunit(grid::HashGrid) = one(inttype(grid))
# linearindices(grid::HashGrid) = grid.lininds
# cartesianindices(grid::HashGrid) = CartesianIndices(grid.gridsize)
floattype(grid::HashGrid) = eltype(grid.cellwidth)
numthreads(grid::HashGrid) = unval(grid.nthreads)

# access cells 

Base.size(grid::HashGrid) = grid.gridsize
Base.length(grid::HashGrid) = prod(size(grid))


function celldomain(grid::HashGrid, i)
    one_ = oneunit(grid)
    gridind = hash2pos(grid, i) # Tuple(cartesianindices(grid)[i])
    return (grid.origin .+ grid.cellwidth .* (gridind .- one_), grid.origin .+ grid.cellwidth .* gridind)
end 

cellsize(grid::HashGrid, i) = grid.cellwidth

@inline Base.getindex(grid::HashGrid, k::Integer) = view(grid.pointidx, grid.cellstarts[k]:grid.cellends[k])

@inline function Base.getindex(grid::HashGrid, i...) 
    k = grid2hash(grid, i)
    return grid[k]
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
    IntT = eltype(gridsize)

    cellwidth = SVector(ntuple(i -> @compat(get(radius, i, radius[end])), Dim))
    cellwidthinv = one(FltT) ./ cellwidth

    cellidx  = IndexVecT(undef, npts)
    pointidx = IndexVecT(undef, npts)

    ncells = prod(gridsize)
    cellstarts = IndexVecT(undef, ncells)
    cellends   = IndexVecT(undef, ncells)

    strides = cumprod(gridsize)
    strides = IntT.((sum(strides[1:end-1]), strides[1:end-1]...))
    
    return HashGrid(cellwidth, cellwidthinv, gridsize, origin, cellidx, pointidx, cellstarts, cellends, strides, backend, Val(nthreads))
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

@inline function pos2grid_unbound(grid, pos)
    IntT = inttype(grid)
    ind = @. ceil(IntT, (pos - grid.origin) * grid.cellwidthinv)
    return Tuple(ind)
end

@inline function pos2grid(grid, pos)
    IntT = inttype(grid)
    ind = @. mod1(ceil(IntT, (pos - grid.origin) * grid.cellwidthinv), grid.gridsize)
    return Tuple(ind)
end

@inline function grid2hash(grid, ind) 
    return ind[1] + sum(grid.strides[2:end] .* ind[2:end]) - grid.strides[1]
end
@inline pos2hash(grid, pos) = grid2hash(grid, pos2grid(grid, pos))

@inline function hash2pos(grid, hash::Tuple)
    return hash
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
    if tid == one(tid)
        grid.cellstarts[c] = one(tid)
    else
        p = grid.cellidx[tid-one(tid)]
        if c != p
            grid.cellstarts[c] = tid 
            grid.cellends[p] = tid-one(tid)
        end
    end

    if tid == length(grid.pointidx)
        grid.cellends[c] = length(grid.pointidx)
    end
end

function compute_cell_offsets!(grid, nthreads = grid.nthreads)
    compute_cell_offsets_kernel!(grid.backend, unval(nthreads))(grid, ndrange = length(grid.cellidx))
end

function paired_sort!(ix, a)
    sortperm!(ix, a)
    permute!(a, ix)
end

function updatecells!(grid, pts; modul = Base, alg = nothing, nthreads = grid.nthreads)    

    resize!(grid.cellidx, length(pts))
    resize!(grid.pointidx, length(pts))

    fill!(grid.cellstarts, one(inttype(grid)))
    fill!(grid.cellends,  zero(inttype(grid)))

    compute_cell_hashes!(grid, pts, nthreads)
    paired_sort!(grid.pointidx, grid.cellidx)
    compute_cell_offsets!(grid, nthreads)

    # after this operation, pointidx[i] is a point index and cellidx[i] is the cell index containing the point 
    # cellstarts and cellends defines the range of a cell inside 'pointidx' 

    KernelAbstractions.synchronize(grid.backend)
end



# iteration over neighbours 

struct HashGridQuery{HG <: HashGrid, IndT}
    grid::HG
    # cellindices::CIndsT
    starts::IndT
    ends::IndT
end
Base.IteratorSize(::HashGridQuery) = Base.SizeUnknown()
Base.eltype(query::HashGridQuery) = eltype(query.grid.pointidx) 


function HashGridQuery(grid::HashGrid, pos, r)
    starts =     pos2grid_unbound(grid, pos .- r)
    ends   = min.(pos2grid_unbound(grid, pos .+ r), starts .+ grid.gridsize .- oneunit(grid))
    # cellindices = starts:ends
    
    return HashGridQuery(grid, starts, ends)
end

function warplinear(grid, ind) 
    return grid2hash(grid, mod1.(ind, grid.gridsize))
end 

function Base.iterate(query::HashGridQuery)
    cellind = query.starts
    linearidx = warplinear(query.grid, cellind)
    i       = query.grid.cellstarts[linearidx]
    cellend = query.grid.cellends[linearidx]

    initstate = (cellind, i, cellend)

    return iterate(query, initstate)
end

@inline function iteratemultiindex(starts, ends, cellind)
    # IntT = eltype(starts)

    # c_1 = cellind[1]
    # if c_1 < ends[1]
    #     cellind = (c_1 + one(IntT), cellind[2], cellind[3])
    # else 
    #     c_1 = starts[1]
    #     c_2 = cellind[2]
    #     if c_2 < ends[2]
    #         cellind = (c_1, c_2 + one(IntT), cellind[3])
    #     else
    #         c_2 = starts[2]
    #         c_3 = cellind[3]
    #         if c_3 < ends[3]
    #             cellind = (c_1, c_2, c_3 + one(IntT))
    #         else
    #             return nothing
    #         end                        
    #     end
    # end


    for k in eachindex(starts)
        c_k = cellind[k] + one(eltype(starts))

        if c_k <= ends[k]
            return Base.setindex(cellind, c_k, k)
        else
            cellind = Base.setindex(cellind, starts[k], k)
        end
    end
    return nothing
end

function Base.iterate(query::HashGridQuery, state)
    (cellind, i, cellend) = state
    IntT = inttype(query.grid)

    @inbounds while true
        if i <= cellend
            k = query.grid.pointidx[i]
            return (k, (cellind, i + one(IntT), cellend))
        else
            cellind = iteratemultiindex(query.starts, query.ends, cellind)
            
            if isnothing(cellind)
                return nothing 
            end

            linearidx = warplinear(query.grid, cellind)
            i       = query.grid.cellstarts[linearidx]
            cellend = query.grid.cellends[linearidx]
        end
    end 
end