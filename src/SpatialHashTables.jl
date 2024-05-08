module SpatialHashTables

using StaticArrays
using KernelAbstractions
import Adapt: adapt_structure

# elementary definitions
unval(::Val{x}) where {x} = x
unval(x) = x

include("core.jl")
include("adapt.jl")

export HashGrid, HashGridQuery
export updatecells!, neighbours

end
