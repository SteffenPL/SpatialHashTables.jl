using SpatialHashTables: numthreads, domainsize

force_fnc(Xi, Xj, Xij, d²) = Xij / sqrt(d²)

function forces_naive!(F, X, r, period = @SVector[1.0, 1.0, 1.0])
    periodinv = 1 ./ period

    for i in eachindex(X)
        Xi = X[i]
        Fi = zero(SVec3d)
        for j in eachindex(X)
            Xj = X[j]
            Xij = warp_vec(Xi - Xj, period, periodinv)
            d² = dist_sq(Xij)
            if 0 < d² < r^2
                Fi += force_fnc(Xi, Xj, Xij, d²)
            end 
        end
        F[i] = Fi
    end
    return nothing
end

function forces_singlecore!(F, X, r, grid)
    period = domainsize(grid)
    periodinv = 1 ./ period

    for i in eachindex(X)
        Xi = X[i]
        Fi = zero(SVec3d)
        for j in HashGridQuery(grid, Xi, r)
            Xj = X[j]
            Xij = warp_vec(Xi - Xj, period, periodinv)
            d² = dist_sq(Xij)
            if 0 < d² < r^2
                Fi += force_fnc(Xi, Xj, Xij, d²)
            end 
        end
        F[i] = Fi
    end
    return nothing
end

@kernel function forces_multithreaded_kernel!(F, @Const(X), r, grid, period, periodinv)
    i = @index(Global)
    
    Xi = X[i]
    Fi = zero(SVec3d)

    query = HashGridQuery(grid, X[i], r)
    for j in query
        Xj = X[j]

        Xij = warp_vec(Xi - Xj, period, periodinv)
        d² = dist_sq(Xij)
        if 0 < d² < r^2
            Fi += force_fnc(Xi, Xj, Xij, d²)
        end 
    end
    F[i] = Fi    
end

function forces_multithreaded!(F, X, r, grid)
    period = domainsize(grid)
    periodinv = 1 ./ period

    kernel = forces_multithreaded_kernel!(grid.backend, numthreads(grid))
    kernel(F, X, r, grid, period, periodinv, ndrange = length(X))

    synchronize(grid.backend)
    return nothing
end

function force_clm(r,x,y,i,j,d²,F)
    if 0 < d² < r^2
        Fij = force_fnc(x, y, x-y, d²)
        F[i] += Fij
        F[j] -= Fij
    end 
    return F
end

function forces_celllistmap!(r, system)
    map_pairwise!((x,y,i,j,d²,F) -> force_clm(r,x,y,i,j,d²,F), system)
    return nothing
end



# primiary functions to generate timing:
function append_data(df, name, N, bench, show_results)
    df2 = DataFrame(
        method = name, 
        N = N,
        time_median = median(bench.times) / 1e9,
        time_mean = mean(bench.times) / 1e9,
        time_std = std(bench.times) / 1e9,
        gc_median = median(bench.gctimes) / 1e9,
        memory_mb = bench.memory / 1024^2,
        allocs = bench.allocs,
        n_runs = length(bench.times)
    )

    if show_results
        printstyled("Runtime for $(name):", color = :green)
        print("\t$(df2.time_median) [sec]\n")
    end

    return vcat(df, df2, cols = :union)
end

function forcebenchmark(N, ptsgen, 
                            name;
                            naive = true, 
                            serial = true, 
                            gpu = true, 
                            celllistmap = true, 
                            show_results = true)

    df = DataFrame()

    # CPU setup
    X, bounds, r = ptsgen(N)
    grid = HashGrid(X, bounds..., r)

    # CellListMap setup 
    system = ParticleSystem(
        xpositions = X, 
        unitcell = bounds[2],
        cutoff = r, 
        output = similar(X),
        output_name = :forces
    )

    # GPU setup 
    if gpu
        X_gpu = cu(SVec3f.(X))
        r_gpu = Float32(r)
        bounds_gpu = SVec3f.(bounds)

        grid_gpu = HashGrid{CUDA.CuVector{Int32}}(X_gpu, bounds_gpu..., r_gpu; nthreads = 256)
    end

    F_n = similar(X)
    F_s = similar(X)
    F_p = similar(X)
    F_gpu = similar(X_gpu)
    
    if show_results
        printstyled("\nStart benchmarks for N = $(N)\n", color = :blue)
    end 

    res_cpu = @benchmark forces_multithreaded!($F_p, $X, $r, $grid)
    df = append_data(df, "CPU (parallel)", N, res_cpu, show_results)

    if serial
        forces_singlecore!(F_s, X, r, grid)
        res_single = @benchmark forces_singlecore!($F_s, $X, $r, $grid)
        df = append_data(df, "CPU (serial)  ", N, res_single, show_results)
        @assert F_p ≈ F_s 
    end

    if gpu 
        forces_multithreaded!(F_gpu, X_gpu, r_gpu, grid_gpu)
        res_gpu = @benchmark forces_multithreaded!($F_gpu, $X_gpu, $r_gpu, $grid_gpu)
        df = append_data(df, "GPU           ", N, res_gpu, show_results)
        @assert norm(SVec3f.(F_p) - Vector(F_gpu)) < 0.1 * norm(F_p)
    end

    if naive && N <= 1_000
        forces_naive!(F_n, X, r)
        res_naive = @benchmark forces_naive!($F_n, $X, $r)
        df = append_data(df, "naive         ", N, res_naive, show_results)
        @assert F_p ≈ F_n
    end 

    if celllistmap
        forces_celllistmap!(r, system)
        res_celllistmap = @benchmark forces_celllistmap!($r, $system)
        df = append_data(df, "CellListMap   ", N, res_celllistmap, show_results)
        @assert F_p ≈ system.forces
    end
    
    df[!, :name] = fill(name, length(df.N))

    return df 
end
