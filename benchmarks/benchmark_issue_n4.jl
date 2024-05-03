using Revise
using StaticArrays
using SpatialHashTables
using CellListMap
using Test
using BenchmarkTools
using ChunkSplitters
using Base.Threads
using LinearAlgebra

using SpatialHashTables: gridindices, hashindex

dist_sq(x,y) = sum(abs2, x - y)

function energy(x, y, i, j, d2, u, cutoff)
    if d2 < cutoff^2
        u += 1.0
        u += 1e-9 * dist_sq(x,y)
    end
    return u 
end


x, _ = CellListMap.xatomic(10^5)
box = Box(limits(x), 12.0)
cl = CellList(x, box)

N = length(x)
r = box.cutoff
Dim = length(eltype(x))
X = x
bht = BoundedHashTable(X, r, limits(x).limits .+ 12 ) # this errors
sht = SpatialHashTable(X, r, 730)







function batchj(irange, ht, X, r)
    ei = 0.0
    for i in irange 
        Xi = X[i]
        for j in neighbours(ht, X[i], r)
            if i < j
                ei = energy(Xi, X[j], i, j, dist_sq(X[i], X[j]), ei, r)
            end
        end
    end
    return ei
end

using FLoops

function test_parallel_3(ht, X, r)
    nchunks = Threads.nthreads()
    e = zeros(nchunks)
    
    Threads.@threads for i in eachindex(X)
        tid = Threads.threadid() 
        ei = e[tid]
        Xi = X[i]
        for j in neighbours(ht, Xi, r)
            if i < j 
                ei = energy(Xi, X[j], i, j, dist_sq(Xi, X[j]), ei, r)
            end
        end
        e[tid] = ei 
    end
    return sum(e)
end

using OhMyThreads
function test_parallel(ht, X, r)

    e = @tasks for i in eachindex(X)
        @set reducer = +

        ei = 0.0
        Xi = X[i]
        for j in neighbours(ht, Xi, r)
            if i < j
                Xj = X[j] 
                ei = energy(Xi, Xj, i, j, dist_sq(Xi, Xj), ei, r)
            end
        end

        ei 
    end

    return e
end
@time test_parallel(bht,  X, r)

@btime test_parallel($bht,  $X, $r)
@time test_parallel(bht,  X, r)
@time test_parallel(sht,  X, r)
# @time test_parallel_(bht,  X, r)
@time test_parallel_2(bht,  X, r)
@time run_celllistmap(box, cl)

x, _ = CellListMap.xatomic(10^5)
box = Box(limits(x), 12.0)
cl = CellList(x, box)

N = length(x)
r = box.cutoff
Dim = length(eltype(x))
X = x
bht = BoundedHashTable(X, r, limits(x).limits .+ 12 ) # this errors
sht = SpatialHashTable(X, r, 730)

@test test_parallel(bht, X, r) ≈ map_pairwise!((x,y,i,j,d2,u) -> energy(x,y,i,j,d2,u,box.cutoff), 0.0, box, cl)
@test test_parallel(sht, X, r) ≈ map_pairwise!((x,y,i,j,d2,u) -> energy(x,y,i,j,d2,u,box.cutoff), 0.0, box, cl)

@btime test_parallel_($bht, $X, $r) 

@btime test_parallel($sht, $X, $r) 

function run_celllistmap(box,cl)
    u = CellListMap.map_pairwise!(
        (x,y,i,j,d2,u) -> energy(x,y,i,j,d2,u,box.cutoff), 
        0.0, box, cl
    ) 
    return u
end
@btime run_celllistmap($box, $cl)


function test_serial(ht, X, r)
    e = 0.0
    for i in eachindex(X) 
        for j in neighbours(ht, X[i], r)
            if i < j
                e = energy(X[i], X[j], i, j, dist_sq(X[i], X[j]), e, r)
            end
        end
    end
    return e
end

run_celllistmap(box, cl) 
@btime test_serial($bht, $X, $r) 
test_parallel(bht, X, r)

@profview test_serial(bht, X, r) 




function batch(ht, X, i, Xi, neighbour_indices, e, r)
    gridpos = gridindices(ht, Xi)
    for boxoffset in neighbour_indices

        boxrep = gridpos .+ Tuple(boxoffset)
        if all(@. 1 <= boxrep <= ht.gridsize)
            boxhash = sum(@. (boxrep - 1) * ht.strides) + 1

            box_start = ht.cellcount[boxhash] + 1 
            box_end = ht.cellcount[boxhash+1]
            for k in box_start:box_end
                j = ht.particlemap[k]
                Xj = X[j]
                if i < j 
                    e = energy(Xi, Xj, i, j, dist_sq(Xi, Xj), e, r)
                end
            end
        end
    end
    return e
end

function test_parallel_(ht, X, r)
    nchunks = Threads.nthreads()
    e = zeros(nchunks)
    Dim = Val(3)
    widths = @. ceil(Int, r * ht.inv_cellsize)
    neighbour_indices = CartesianIndices(ntuple(i -> -widths[i]:widths[i], Dim))

    Threads.@threads for i in eachindex(X)
        ichunk = Threads.threadid()
        Xi = X[i]
        e[ichunk] = batch(ht, X, i, Xi, neighbour_indices, e[ichunk], r)
    end
    return sum(e)
end

function test_parallel_2(ht, X, r)
    nchunks = Threads.nthreads()
    e = zeros(nchunks)
    Dim = Val(3)
    widths = @. ceil(Int, r * ht.inv_cellsize)
    neighbour_indices = CartesianIndices(ntuple(i -> -widths[i]:widths[i], Dim))

    Threads.@threads for i in eachindex(X)
        ichunk = Threads.threadid()
        ei = e[ichunk]
        Xi = X[i]
        gridpos = gridindices(ht, Xi)
        for boxoffset in neighbour_indices
            boxrep = gridpos .+ Tuple(boxoffset)
            if all(@. 1 <= boxrep <= ht.gridsize)
                boxhash = sum(@. (boxrep - 1) * ht.strides) + 1

                box_start = ht.cellcount[boxhash] + 1 
                box_end = ht.cellcount[boxhash+1]
                for k in box_start:box_end
                    j = ht.particlemap[k]
                    Xj = X[j]
                    if i < j 
                        ei = energy(Xi, Xj, i, j, dist_sq(Xi, Xj), ei, r)
                    end
                end
            end
        end
        e[ichunk] = ei
    end
    return sum(e)
end

@time test_parallel(bht,  X, r) 
@time test_parallel_(bht,  X, r) 
@time test_parallel_2(bht, X, r) 

# 0.824237 seconds (253 allocations: 35.656 KiB)
# 0.414048 seconds (243 allocations: 35.391 KiB)

1

# function batchi(ht, X, irange, neighbour_indices, e, ichunk, r)
#     ei = e[ichunk]
#     for i in irange 
#         Xi = X[i]
#         ei = batch(ht, X, i, Xi, neighbour_indices, ei, r)
#     end
#     e[ichunk] = ei
#     return nothing
# end

# function test_parallel_(ht, X, r)
#     nchunks = Threads.nthreads()
#     e = zeros(nchunks)
#     Dim = Val(3)
#     # widths = @. ceil(Int, r * ht.inv_cellsize)
#     neighbour_indices = CartesianIndices(ntuple(i -> -1:1, Dim))

#     @sync for (irange, ichunk) in chunks(X, nchunks)
#         @spawn batchi($ht, $X, $irange, $neighbour_indices, $e, $ichunk, $r)
#     end
#     return sum(e)
# end







using JET


@time test_parallel_(bht, X, r) 
# 0.824237 seconds (253 allocations: 35.656 KiB)
# 0.414048 seconds (243 allocations: 35.391 KiB)
#
@report_opt test_parallel_(bht, X, r)

@time test_parallel_(bht, X, r) 
@profview test_parallel_(bht, X, r)



using BenchmarkTools

function naive(N)
    x = ones(N)
    r = 1.0
    e = zeros(Threads.nthreads())

    Threads.@threads for i in eachindex(x)
        tid = Threads.threadid()
        ei = e[tid]
        for j in eachindex(x) 
            ei += x[i]*x[j]*r 
        end
        e[tid] = ei
    end
    return sum(e)
end

function batch_ex(x, i, tid, e, r)
    ei = e[tid]
    for j in eachindex(x) 
        ei += x[i]*x[j]*r
    end
    e[tid] = ei
end

function with_func_barrier(N)
    x = ones(N)
    r = 1.0

    e = zeros(Threads.nthreads())


    Threads.@threads for i in eachindex(x)
        tid = Threads.threadid()
        batch_ex(x, i, tid, e, r) 
    end
    return sum(e)
end

@time naive(10000)
@time with_func_barrier(10000)



# using BenchmarkTools
# function benchmark_settype(::Type{S}; universe_size, set_size) where {S}
#     @btime union(A, B) setup = ( #
#         A = $S(rand(1:$universe_size, $set_size));
#         B = $S(rand(1:$universe_size, $set_size))
#     )
#     return nothing
# end

# function quick_union(A, B)
#     a, b = A[1], B[1]
#     iA, iB = 1, 1
#     C = union(a,b)
#     sort!(C)
#     new_a = a 
#     last_a = a 

#     new_b = b 
#     last_b = b

#     @inbounds while true
#         last_four = (new_a, last_a, new_b, last_b)

#         if a <= b && iA < length(A)
#             iA += 1
#             a = A[iA]
#             if !(a in last_four)
#                 push!(C, a)
#                 last_a, new_a = new_a, a
#             end
#         elseif iB < length(B)
#             iB += 1
#             b = B[iB]
#             if !(b in last_four)
#                 push!(C, b)
#                 last_b, new_b = new_b, b
#             end
#         else 
#             break 
#         end
#     end

#     return C 
# end



# struct SortedVector{T}
#     data::T

#     function SortedVector{T}(x::Vector) where {T}
#         new{Vector{T}}(convert.(T,sort(x)))
#     end
# end
# Base.union(A::SortedVector, B::SortedVector) = quick_union(A,B)
# Base.getindex(A::SortedVector, i) = A.data[i]
# Base.length(A::SortedVector) = length(A.data)

# struct NonPreSortedVector{T}
#     data::T
#     function NonPreSortedVector{T}(x::Vector) where {T}
#         new{Vector{T}}(convert.(T,x))
#     end
# end
# function Base.union(A::NonPreSortedVector, B::NonPreSortedVector) 
#     sort!(A.data)
#     sort!(B.data)
#     quick_union(A,B)
# end

# Base.getindex(A::NonPreSortedVector, i) = A.data[i]
# Base.length(A::NonPreSortedVector) = length(A.data)


# benchmark_settype(Set{UInt}; universe_size = 10^6, set_size = 10^2)
# benchmark_settype(BitSet; universe_size=10^6, set_size=10^2)
# benchmark_settype(SortedVector{UInt}; universe_size = 10^6, set_size = 10^2)
# benchmark_settype(NonPreSortedVector{UInt}; universe_size = 10^6, set_size = 10^2)

# benchmark_settype(Set{UInt}; universe_size = 10^6, set_size = 10^3)
# benchmark_settype(BitSet; universe_size=10^6, set_size=10^3)
# benchmark_settype(SortedVector{UInt}; universe_size = 10^6, set_size = 10^3)
# benchmark_settype(NonPreSortedVector{UInt}; universe_size = 10^6, set_size = 10^3)

struct MyIntSet{T}
    n::Nothing
    v::Vector{T}
end

function MyIntSet{T}(v::Vector{S}) where {T,S}
    if T== S
MyIntSet{T}(nothing, sort!(v))
else
MyIntSet{T}(nothing, sort!(convert.(T, v)))
end
end

function Base.union(l::MyIntSet{T}, r::MyIntSet{T}) where T
    left = l.v
    right = r.v
    res = Vector{T}(undef, length(left) + length(right))
    li = 1
    ri = 1
    oi = 1
    until = min(length(left), length(right))
    @inbounds while li <= until && ri <= until
    litem = left[li]
    ritem = right[ri]
    ls = litem <= ritem
    rs = litem >= ritem
    res[oi] = ifelse(ls, litem, ritem)
    oi += 1
    li = ifelse(ls, li+1, li)
    ri = ifelse(rs, ri+1, ri)
    end
    @inbounds for i=li:length(left)
    res[oi] = left[i]
    oi += 1
    end
    @inbounds for i=ri:length(right)
    res[oi] = right[i]
    oi += 1
    end
    resize!(res, oi-1)
    MyIntSet(nothing, res)
    end

A = rand(1:1000, 500)
B = rand(1:1000, 500)
sort!(A)
sort!(B)
sort(union(MyIntSet(nothing, A),MyIntSet(nothing, B)).v) == sort(union(A,B))


function test_parallel_2(ht, X, r)
    nchunks = Threads.nthreads()
    e = zeros(nchunks)
    Dim = Val(3)
    widths = @. ceil(Int, r * ht.inv_cellsize)
    neighbour_indices = CartesianIndices(ntuple(i -> -widths[i]:widths[i], Dim))

    Threads.@threads for i in eachindex(X)
        ichunk = Threads.threadid()
        ei = e[ichunk]
        Xi = X[i]
        gridpos = gridindices(ht, Xi)
        for boxoffset in neighbour_indices
            boxrep = gridpos .+ Tuple(boxoffset)
            if all(@. 1 <= boxrep <= ht.gridsize)
                boxhash = sum(@. (boxrep - 1) * ht.strides) + 1
                box_start = ht.cellcount[boxhash] + 1 
                box_end = ht.cellcount[boxhash+1]
                for k in box_start:box_end
                    j = ht.particlemap[k]
                    Xj = X[j]
                    if i < j 
                        ei = energy(Xi, Xj, i, j, dist_sq(Xi, Xj), ei, r)
                    end
                end
            end
        end
        e[ichunk] = ei
    end
    return sum(e)
end


function NI(ht, r) 

end 

ni = NeighbourIterator(ht, r)

for i in eachindex(X)
    Xi = X[i]
    for j in ni(Xi)
        # do something
    end
end
