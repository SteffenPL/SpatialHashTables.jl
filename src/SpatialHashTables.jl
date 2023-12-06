module SpatialHashTables
using StaticArrays
import Adapt: adapt_structure

include("abstractspatialhashtables.jl")
include("boundedhashtable.jl")
include("spatialhashtable.jl")
include("core.jl")
include("adapt.jl")

export SpatialHashTable, BoundedHashTable, AbstractSpatialHashTable
export updatetable!, resize!, neighbours, iterate_box, dimension, inside, hashposition, periodic_neighbours
end
