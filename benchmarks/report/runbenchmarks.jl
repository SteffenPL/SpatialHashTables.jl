include("common.jl")
include("systeminfo.jl")
include("forcebenchmark.jl")


# create a new folder for the results 
basename = joinpath("report", "output", string(Dates.today()))
mkpath(basename)

collectsysteminfo(joinpath(basename, "sysinfo"))


# BenchmarkTools.DEFAULT_PARAMETERS.samples = 10

# Case 1: Uniform points with fixed number of particles per ideal inner loop

function example_uniform(N; seed = nothing)
    r = 1/N^(1/3)

    if !isnothing(seed)
        Random.seed!(seed)
    end

    X = rand(SVector{3, Float64}, N)

    return X, (@SVector[0.0,0.0,0.0], @SVector[1.0,1.0,1.0]), r
end

# using CSV
# X, bnds, r = example_uniform(1_000_000, seed = 1)
# X_ = cu(SVec3f.(X))
# bnds_ = SVec3f.(bnds)
# r_ = Float32(r)
# F_ = similar(X_)

# grid_ = HashGrid{CuArray{Int32}}(X_, bnds_..., r_; nthreads = 32)
# @time forces_multithreaded!(F_, X_, r_, grid_)

# F_py = reshape(reinterpret(Float32, Array(F_)), 3, :)
# X_py = reshape(reinterpret(Float32, Array(X_)), 3, :)
# using NPZ

# npzwrite("example_uniform.npz", Dict("points" => X_py', "forces" => F_py', "radius" => r_))

df_unif1 = forcebenchmark(      100, example_uniform, "uniform")
df_unif2 = forcebenchmark(    1_000, example_uniform, "uniform")
df_unif3 = forcebenchmark(   10_000, example_uniform, "uniform"; naive = false)
df_unif4 = forcebenchmark(  100_000, example_uniform, "uniform"; naive = false)
df_unif5 = forcebenchmark(1_000_000, example_uniform, "uniform"; naive = false, serial = false)
# or use 
# dfs = (forcebenchmark(N, example_uniform, "uniform"; naive = N <= 10_000, serial = N <= 100_000)
#            for N in 10 ^ 2:6)

df_unif = vcat(df_unif1, df_unif2, df_unif3, df_unif4, df_unif5)
# df_unif = vcat(dfs...)

CSV.write(joinpath(basename, "uniform.csv"), df_unif)

@df df_unif plot(:N, :time_median, 
                group = (:method), 
                xaxis=:log, 
                yaxis=:log,
                legend = :topleft,
                linewidth = 2,
                yticks = 10.0 .^ (-5:2),
                xlabel = "N",
                ylabel = "seconds (median)",
                title = "Uniform random particles with r = 1/N³",
                palette = :tab10)

savefig(joinpath(basename, "uniform.png"))

# Case 2
function example_atomic(N; seed = nothing)

    if !isnothing(seed)
        Random.seed!(seed)
    end

    X, box = CellListMap.xatomic(N)
    r = box.cutoff
    upperbound = diag(box.aligned_unit_cell.matrix)

    return X, (@SVector[0.0,0.0,0.0], upperbound), r
end


# using CSV
# X, bnds, r = example_atomic(1_000_000, seed = 1)
# X_ = cu(SVec3f.(X))
# bnds_ = SVec3f.(bnds)
# r_ = Float32(r)
# F_ = similar(X_)

# grid_ = HashGrid{CuArray{Int32}}(X_, bnds_..., r_; nthreads = 32)
# @time forces_multithreaded!(F_, X_, r_, grid_)

# F_py = reshape(reinterpret(Float32, Array(F_)), 3, :)
# X_py = reshape(reinterpret(Float32, Array(X_)), 3, :)
# using NPZ

# npzwrite("example_atomic.npz", Dict("points" => X_py', "forces" => F_py', "radius" => r_, "gridsize" => collect(grid_.gridsize)))



# X, b, r = example_atomic(10_000)
# system = ParticleSystem(
#         xpositions = X, 
#         unitcell = b[2],
#         cutoff = r, 
#         output = similar(X),
#         output_name = :forces
#     )

# grid = HashGrid(X, b..., r)

Ns = [3_000, 10_000, 30_000, 100_000, 300_000, 1_000_000]  # 3_000_000]
dfs_atomic = []

for N in Ns
    push!(dfs_atomic,
    forcebenchmark(N, example_atomic, "atomic"; naive = false, serial = false)
    )
end

df_xatomic = vcat(dfs_atomic...)
CSV.write(joinpath(basename, "xatomic.csv"), df_xatomic)

cols = palette(:tab10)
@df df_xatomic plot(:N, :time_median, 
                group = (:method),
                xaxis=:log, 
                yaxis=:log,
                legend = :topleft,
                linewidth = 2,
                yticks = 10.0 .^ (-5:2),
                xlabel = "N",
                ylabel = "seconds (median)",
                title = "CellListMap.xatomic",
                palette = [cols[1], cols[2], cols[4], cols[5]])
            
                
savefig(joinpath(basename, "xatomic.png"))






# Case 3
function example_galactic(N)
    X, box = CellListMap.xgalactic(N)
    r = box.cutoff
    upperbound = diag(box.aligned_unit_cell.matrix)

    return X, (@SVector[0.0,0.0,0.0], upperbound), r
end

# X, b, r = example_atomic(10_000)
# system = ParticleSystem(
#         xpositions = X, 
#         unitcell = b[2],
#         cutoff = r, 
#         output = similar(X),
#         output_name = :forces
#     )

# grid = HashGrid(X, b..., r)

Ns = [30_000, 100_000, 300_000, 1_000_000]
dfs_galactic = []

for N in Ns
    push!(dfs_galactic,
    forcebenchmark(N, example_galactic, "galactic"; naive = N < 10_000, serial = N < 200_000)
    )
end

df_galactic = vcat(dfs_galactic...)
CSV.write(joinpath(basename, "xgalactic.csv"), df_galactic)

@df df_galactic plot(:N, :time_median, 
                group = (:method), 
                xaxis=:log, 
                yaxis=:log,
                legend = :topleft,
                linewidth = 2,
                yticks = 10.0 .^ (-5:2),
                xlabel = "N",
                ylabel = "seconds (median)",
                title = "CellListMap.xgalactic")
            
                
savefig(joinpath(basename, "xgalactic.png"))
