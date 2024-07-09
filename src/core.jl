using Base.Cartesian: @nloops, @ntuple
using KernelAbstractions: @context

abstract type AbstractGrid end 

struct BoundedGrid{Dim, FltT<:Real, IntT<:Integer, IntVecT, BackendT, NThreads} <: AbstractGrid    
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

Base.show(io::IO, grid::BoundedGrid) = print(io, "BoundedGrid (gridsize = ", grid.gridsize, ", cellwidth = ", grid.cellwidth, ", backend = ", grid.backend, ", nthreads = ", unval(grid.nthreads), ")")

# source: https://matthias-research.github.io/pages/publications/tetraederCollision.pdf
const default_factors = (92837111, 689287499, 283923481)

struct HashGrid{Dim, FltT<:Real, IntT<:Integer, IntVecT, BackendT, NThreads <: Val} <: AbstractGrid    
    cellwidth::SVector{Dim,FltT}
    cellwidthinv::SVector{Dim,FltT}

    cellidx::IntVecT 
    pointidx::IntVecT

    cellstarts::IntVecT
    cellends::IntVecT 

    pseudorandom_factors::NTuple{Dim,IntT}

    backend::BackendT
    nthreads::NThreads
end

Base.show(io::IO, grid::HashGrid) = print(io, "HashGrid (numcells = ", length(grid.cellstarts), ", cellwidth = ", grid.cellwidth, ", backend = ", grid.backend, ", nthreads = ", unval(grid.nthreads), ")")


# == Convention == 
#   
# All constructors require the following information:
# 1. cutoff (either scalar or vector; determines float type) 
# 2. number of cells (either defined by: size of a domain (Float64s), number of cells per dimension (Int64s), total number of cells (Int64))
# 3. number of points + dimension (either by number of points + dim, or concrete Vector{SVector})
# 4. optional: type of index vector (e.g. Vector{Int64} or CuVector{Int32}, determines float type)
#
# If a Vector/CuVector of SVectors is provided, updatecells! will be called automatically.

function HashGrid{dim}(cutoff::FT, ncells::Integer, npts::Integer, ::Type{IndexVecT}; backend = KernelAbstractions.CPU(), nthreads = 16) where {dim, FT, IndexVecT <: AbstractVector}
    IT = eltype(IndexVecT)

    cellwidth = SVector(ntuple(i -> get(cutoff, i, cutoff[end]), dim))
    cellwidthinv = FT(1) ./ cellwidth

    cellidx  = IndexVecT(undef, npts)
    pointidx = IndexVecT(undef, npts)

    cellstarts = IndexVecT(undef, ncells)
    cellends   = IndexVecT(undef, ncells)
    
    pseudorandom_factors = IT.(default_factors[1:dim])

    return HashGrid(cellwidth, cellwidthinv, cellidx, pointidx, cellstarts, cellends, pseudorandom_factors, backend, val(nthreads))
end

# # default for CPUs 
HashGrid{dim}(cutoff, ncells::Integer, npts::Integer; kwargs...) where {dim} = HashGrid{dim}(cutoff, ncells, npts, Vector{Int64}; kwargs...)

# define npts and dim with an input vector 
function HashGrid(cutoff, ncells::Integer, pts::AbstractArray, args...; backend = get_backend(pts), kwargs...) where {FT, VT}
    npts = length(pts)
    dim = length(eltype(pts))
    grid = HashGrid{dim}(cutoff, ncells, npts, args...; backend, kwargs...)
    updatecells!(grid, pts)
    return grid
end



function BoundedGrid(cutoff, gridsize::NTuple{dim,T}, npts::Integer, ::Type{IndexVecT}; origin = zero(SVector{dim,eltype(cutoff)}), backend = KernelAbstractions.CPU(), nthreads = 16) where {dim, T <: Integer, IndexVecT <: AbstractVector}
    FT = eltype(cutoff)
    IT = eltype(IndexVecT)
    gridsize = IT.(gridsize)

    cellwidth = SVector(ntuple(i -> @compat(get(cutoff, i, cutoff[end])), dim))
    cellwidthinv = FT(1) ./ cellwidth

    cellidx  = IndexVecT(undef, npts)
    pointidx = IndexVecT(undef, npts)

    ncells = prod(gridsize)
    cellstarts = IndexVecT(undef, ncells)
    cellends   = IndexVecT(undef, ncells)

    strides = cumprod(gridsize)
    strides = IT.((sum(strides[1:end-1]), strides[1:end-1]...))

    return BoundedGrid(cellwidth, cellwidthinv, gridsize, origin, cellidx, pointidx, cellstarts, cellends, strides, backend, val(nthreads))
end

# default for CPU 
function BoundedGrid(cutoff, gridsize::NTuple{dim,T}, npts::Integer; kwargs...) where {dim, T <: Integer}
    return BoundedGrid(cutoff, gridsize, npts, Vector{Int64}; kwargs...)
end

# define pts as a vector
function BoundedGrid(cutoff, gridsize::NTuple{dim,T}, pts::AbstractVector, args...; backend = get_backend(pts), kwargs...) where {dim, T <: Integer}
    npts = length(pts)
    grid = BoundedGrid(cutoff, gridsize, npts, args...; backend, kwargs...)
    updatecells!(grid, pts)
    return grid
end

# define domainsize instead of gridsize
function BoundedGrid(cutoff::FT, domainsize::SVector{dim,FT}, args...; kwargs...) where {dim, FT <: Real}
    gridsize = Tuple(@. ceil(FT == Int32 ? Int32 : Int64, domainsize / cutoff))
    cutoff = FT.(domainsize ./ gridsize)

    return BoundedGrid(cutoff, gridsize, args...; kwargs...)
end


# basic information
dimension(::Type{HashGrid{Dim,FT,IT,IVecT,B,V}}) where {Dim,FT,IT,IVecT,B,V} = Dim
dimension(::Type{BoundedGrid{Dim,FT,IT,IVecT,B,V}}) where {Dim,FT,IT,IVecT,B,V} = Dim
dimension(grid::AbstractGrid) = dimension(typeof(grid))

inttype(::HashGrid{Dim,FT,IT,IVecT,B,V}) where {Dim,FT,IT,IVecT,B,V} = IT
inttype(::BoundedGrid{Dim,FT,IT,IVecT,B,V}) where {Dim,FT,IT,IVecT,B,V} = IT

# linearindices(grid::HashGrid) = grid.lininds
# cartesianindices(grid::HashGrid) = CartesianIndices(grid.gridsize)
floattype(::BoundedGrid{Dim,FT,IT,IVecT,B,V}) where {Dim,FT,IT,IVecT,B,V} = FT
floattype(::HashGrid{Dim,FT,IT,IVecT,B,V}) where {Dim,FT,IT,IVecT,B,V} = FT

numthreads(grid::AbstractGrid) = unval(grid.nthreads)

# access cells 
Base.size(grid::BoundedGrid) = grid.gridsize
Base.size(grid::HashGrid) = length(grid.cellstarts)
Base.length(grid::AbstractGrid) = length(grid.cellstarts)

function celldomain(grid::BoundedGrid, i)
    one_ = inttype(grid)(1)
    gridind = i  # hash2pos(grid, i) # Tuple(cartesianindices(grid)[i])
    return (grid.origin .+ grid.cellwidth .* (gridind .- one_), grid.origin .+ grid.cellwidth .* gridind)
end 

cellsize(grid::AbstractGrid, i) = grid.cellwidth

@inline Base.getindex(grid::AbstractGrid, k::Integer) = view(grid.pointidx, grid.cellstarts[k]:grid.cellends[k])
@inline Base.getindex(grid::AbstractGrid, i...) = getindex(grid, grid2hash(grid, i))

domainsize(grid::BoundedGrid) = size(grid) .* grid.cellwidth
domainsize(grid::HashGrid) = size(grid) .* Inf

domain(grid::BoundedGrid) = (grid.origin, grid.origin + domainsize(grid))
domain(grid::HashGrid) = (-domainsize(grid), domainsize(grid))


# index functions
# - pos  (position, vector)
# - grid (gridindex, cartesian, maybe out of range!) 
# - hash (cellindex, linear)

@inline function pos2grid(grid::BoundedGrid, pos)
    IT = inttype(grid)
    ind = @. ceil(IT, (pos - grid.origin) * grid.cellwidthinv)
    return Tuple(ind)
end

@inline function pos2grid(grid::HashGrid, pos)
    IT = inttype(grid)
    ind = @. ceil(IT, pos * grid.cellwidthinv)
    return Tuple(ind)
end

@inline function grid2hash(grid::BoundedGrid, ind)
    ind_mod = @. mod1(ind, grid.gridsize)
    k = ind_mod[1] + sum(@. grid.strides[2:end] * ind_mod[2:end]) - grid.strides[1]
    return k
end

@inline function grid2hash(grid::HashGrid, ind)
    return mod1(abs(reduce(⊻, ind .* grid.pseudorandom_factors)), length(grid))
end

@inline pos2hash(grid, pos) = grid2hash(grid, pos2grid(grid, pos))


# update functions
@kernel function compute_cell_hashes_kernel!(grid::AbstractGrid, pts)
    tid = @index(Global)
    grid.cellidx[tid] = pos2hash(grid, pts[tid])
    grid.pointidx[tid] = tid 
end

function compute_cell_hashes!(grid::AbstractGrid, pts, nthreads = grid.nthreads)
    compute_cell_hashes_kernel!(grid.backend, unval(nthreads))(grid, pts, ndrange = length(pts))
end

@kernel function compute_cell_offsets_kernel!(grid::AbstractGrid)
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

function compute_cell_offsets!(grid::AbstractGrid, nthreads = grid.nthreads)
    compute_cell_offsets_kernel!(grid.backend, unval(nthreads))(grid, ndrange = length(grid.cellidx))
end

function paired_sort!(ix, a)
    sortperm!(ix, a)
    permute!(a, ix)
end

function updatecells!(grid, pts; nthreads = grid.nthreads)    
    IT = inttype(grid)

    resize!(grid.cellidx, length(pts))
    resize!(grid.pointidx, length(pts))

    fill!(grid.cellstarts, IT(1))
    fill!(grid.cellends,   IT(0))

    compute_cell_hashes!(grid, pts, nthreads)
    paired_sort!(grid.pointidx, grid.cellidx)
    compute_cell_offsets!(grid, nthreads)

    # after this operation, pointidx[i] is a point index and cellidx[i] is the cell index containing the point 
    # cellstarts and cellends defines the range of a cell inside 'pointidx' 

    KernelAbstractions.synchronize(grid.backend)
end



# iteration over neighbours 
@inline function iteratemultiindex(starts, ends, cellind)
    IT = eltype(starts)
    for k in eachindex(starts)
        c_k = cellind[k] + IT(1)

        if c_k <= ends[k]
            return Base.setindex(cellind, c_k, k)
        else
            cellind = Base.setindex(cellind, starts[k], k)
        end
    end
    return nothing
end

# bounded grid, take care of not wrapping around for periodic boundary conditions
struct BoundedGridQuery{HG <: BoundedGrid, IndT}
    grid::HG
    
    starts::IndT
    ends::IndT
end
Base.IteratorSize(::BoundedGridQuery) = Base.SizeUnknown()
Base.eltype(query::BoundedGridQuery) = eltype(query.grid.pointidx) 


function BoundedGridQuery(grid::AbstractGrid, pos, r)
    IT = inttype(grid)
    starts =      pos2grid(grid, pos .- r)
    ends   = min.(pos2grid(grid, pos .+ r), starts .+ grid.gridsize .- IT(1))
    # take min to avoid visiting cells twice
    
    return BoundedGridQuery(grid, starts, ends)
end

function Base.iterate(query::BoundedGridQuery)
    cellind = query.starts
    linearidx = grid2hash(query.grid, cellind)
    i       = query.grid.cellstarts[linearidx]
    cellend = query.grid.cellends[linearidx]

    initstate = (cellind, i, cellend)

    return iterate(query, initstate)
end

function Base.iterate(query::BoundedGridQuery, state)
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

            linearidx = grid2hash(query.grid, cellind)
            i       = query.grid.cellstarts[linearidx]
            cellend = query.grid.cellends[linearidx]
        end
    end 
end


struct HashGridQuery{HG <: HashGrid, VT}
    grid::HG
    hashes::VT
end

Base.IteratorSize(::HashGridQuery) = Base.SizeUnknown()
Base.eltype(query::HashGridQuery) = eltype(query.grid.pointidx) 

function init_hashes(grid::HashGrid{dim}) where {dim}
    IT = inttype(grid)
    return zero(SVector{IT(3)^dim, IT}) # Vector{inttype(grid)}(undef, 3^dimension(grid))
end

@generated function HashGridQuery(grid::HashGrid, pos, r, hashes = init_hashes(grid))
    dim = dimension(grid)
    quote
        IT = inttype(grid)
        FT = floattype(grid)

        @assert r <= maximum(grid.cellwidth)

        k = IT(1)
        @nloops $dim I d->IT(-1):IT(1) begin 

            offset = @ntuple $dim j -> I_j * r
            hash = pos2hash(grid, pos .- offset) 
            isunique = true 
            for l in IT(1):IT(k)-IT(1)
                if hashes[l] == hash 
                    isunique = false
                    break
                end
            end

            if isunique
                hashes = Base.setindex(hashes, hash, k)
                k += IT(1)
            end
        end

        if k <= length(hashes)
            hashes = Base.setindex(hashes, IT(-1), k) 
        end

        return HashGridQuery(grid, hashes)
    end
end


function Base.iterate(query::HashGridQuery)
    IT = inttype(query.grid)

    linearidx = query.hashes[1]
    if linearidx == IT(-1)
        return nothing
    end

    i       = query.grid.cellstarts[linearidx]
    cellend = query.grid.cellends[linearidx]

    initstate = (IT(1), i, cellend)

    return iterate(query, initstate)
end

function Base.iterate(query::HashGridQuery, state)
    (k, i, cellend) = state
    IT = inttype(query.grid)

    @inbounds while true
        if i <= cellend
            j = query.grid.pointidx[i]
            return (j, (k, i + IT(1), cellend))
        else
            k += IT(1) 

            if k > length(query.hashes)
                return nothing 
            end

            cellidx = query.hashes[k]
            if cellidx <= IT(0)
                return nothing 
            end

            i = query.grid.cellstarts[cellidx]
            cellend = query.grid.cellends[cellidx]
        end
    end 
end

# uniform interface to access both types of iterators
neighbours(grid::BoundedGrid, pos, r, hashes = nothing) = BoundedGridQuery(grid, pos, r)
neighbours(grid::HashGrid, pos, r, args...) = HashGridQuery(grid, pos, r, args...)