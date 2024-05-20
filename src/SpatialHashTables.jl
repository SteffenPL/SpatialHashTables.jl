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

export HashGrid, HashGridQuery
export updatecells!, neighbours

end
