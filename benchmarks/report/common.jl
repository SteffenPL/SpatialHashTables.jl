using Revise
using LinearAlgebra, Random, Dates, TOML, Statistics

using StaticArrays
using KernelAbstractions
using DataFrames
using CSV
using BenchmarkTools
using Plots
using StatsPlots

using CellListMap
using SpatialHashTables
using SpatialHashTables: numthreads, domainsize

import CUDA
using CUDA: cu, CuArray

const SVec3d = SVector{3,Float64}
const SVec3f = SVector{3,Float32}

dist_sq(xy) = sum(z -> z^2, xy)

function warp_vec(v, period, periodinv = 1 ./ period)
    return @. v - period * round(v * periodinv)
end
