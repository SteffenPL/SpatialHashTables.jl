include("common.jl")
include("systeminfo.jl")
include("forcebenchmark.jl")


# create a new folder for the results 
basename = joinpath("report", "output", string(Dates.today()))
mkpath(basename)

collectsysteminfo(joinpath(basename, "sysinfo"))




# Case 1: Uniform points with fixed number of particles per ideal inner loop

function example_uniform(N)
    r = 1/N^(1/3)
    X = rand(SVector{3, Float64}, N)

    return X, r, (@SVector[0.0,0.0,0.0], @SVector[1.0,1.0,1.0])
end

BenchmarkTools.DEFAULT_PARAMETERS.samples = 10

df_unif1 = forcebenchmark(      100, example_uniform, "uniform")
df_unif2 = forcebenchmark(    1_000, example_uniform, "uniform")
df_unif3 = forcebenchmark(   10_000, example_uniform, "uniform")
df_unif4 = forcebenchmark(  100_000, example_uniform, "uniform"; naive = false)
df_unif5 = forcebenchmark(1_000_000, example_uniform, "uniform"; naive = false, serial = false)
# or use 
# dfs = (forcebenchmark(N, example_uniform, "uniform"; naive = N <= 10_000, serial = N <= 100_000)
#            for N in 10 ^ 2:6)

df_unif = vcat(df_unif1, df_unif2, df_unif3, df_unif4, df_unif5)
df_unif = vcat(dfs...)

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
                title = "Uniform random particles with r = 1/N³")


# Case 2
function example_atomic(N)
    X, box = CellListMap.xatomic(N)
    r = box.cutoff
    upperbound = diag(box.aligned_unit_cell.matrix)

    return X, r, (@SVector[0.0,0.0,0.0], upperbound)
end

df_atom2 = forcebenchmark(    5_000, example_atomic, "uniform")
df_atom3 = forcebenchmark(   10_000, example_atomic, "uniform")
df_atom4 = forcebenchmark(  100_000, example_uniform, "uniform"; naive = false, serial = false)
df_atom5 = forcebenchmark(1_000_000, example_uniform, "uniform"; naive = false, serial = false)

df_xatomic = vcat(df_atom2,df_atom3,df_atom4)#,df_atom5)
CSV.write(joinpath(basename, "xatomic.csv"), df_xatomic)

@df df_xatomic plot(:N, :time_median, 
                group = (:method), 
                xaxis=:log, 
                yaxis=:log,
                legend = :topleft,
                linewidth = 2,
                yticks = 10.0 .^ (-5:2),
                xlabel = "N",
                ylabel = "seconds (median)",
                title = "CellListMap.xatomic")
                
