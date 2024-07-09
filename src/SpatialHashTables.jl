module SpatialHashTables

using StaticArrays
using KernelAbstractions
import Adapt: adapt_structure
using Compat

# elementary definitions
unval(::Val{x}) where {x} = x
unval(x) = x
val(x) = Val(x) 
val(x::Val) = x

include("core.jl")
include("adapt.jl")
include("operations.jl")

export HashGrid, BoundedGrid, HashGridQuery, BoundedGridQuery 
export updatecells!, neighbours

# optional helpers which one might want to import to handle periodic or non-periodic boundaries

end
