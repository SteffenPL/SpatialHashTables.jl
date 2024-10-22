using SpatialHashTables: numthreads, domainsize

force_fnc(Xi, Xj, Xij, d²) = Xij / sqrt(d²)

function forces_naive!(F, X, r, period = @SVector[1.0, 1.0, 1.0])
    periodinv = 1 ./ period

    for i in eachindex(X)
        Xi = X[i]
        Fi = zero(eltype(F))
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
        for j in neighbours(grid, Xi, r)
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
    Fi = zero(eltype(F))
    r² = r*r

    for j in neighbours(grid, Xi, r)
        Xj = X[j]

        Xij = warp_vec(Xi - Xj, period, periodinv)
        d² = dist_sq(Xij)
        if zero(r) < d² < r²
            Fi += force_fnc(Xi, Xj, Xij, d²)
        end 
    end
    F[i] = Fi    
end


function forces_multithreaded!(F, X, r, grid)
    period = domainsize(grid)
    periodinv = one(eltype(period)) ./ period

    kernel = forces_multithreaded_kernel!(grid.backend, numthreads(grid))
    kernel(F, X, r, grid, period, periodinv, ndrange = length(X))

    synchronize(grid.backend)
    return nothing
end

@kernel function forces_multithreaded_kernel_np!(F, X, r, grid)
    i = @index(Global)
    
    Xi = X[i]
    Fi = zero(eltype(F))
    for j in neighbours(grid, Xi, r)
        Xij = Xi - X[j]
        d = sqrt(dist_sq(Xij))
        if zero(d) < d < r
            Fi += Xij ./ d
        end 
    end
    F[i] = Fi    
end

function forces_multithreaded_np!(F, X, r, grid)

    kernel = forces_multithreaded_kernel_np!(grid.backend, numthreads(grid))
    kernel(F, X, r, grid, ndrange = length(X))

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
                        serial = false,
                        gpu = true,
                        hash_gpu = false,
                        hash_cpu = true,
                        celllistmap = true, 
                        show_results = true)                                

    df = DataFrame()

    # CPU setup
    X, bounds, r = ptsgen(N)
    grid = BoundedGrid(r, bounds[2], X; nthreads = Val(12))
    hgrid = HashGrid(r, length(X), X; nthreads = Val(12))

    # CellListMap setup 
    system = ParticleSystem(
        xpositions = X, 
        unitcell = bounds[2],
        cutoff = r, 
        output = similar(X),
        output_name = :forces
    )

    # GPU setup 
    X_gpu = cu(SVec3f.(X))
    r_gpu = Float32(r)
    bounds_gpu = SVec3f.(bounds)
    
    if gpu || hash_gpu

        grid_gpu = BoundedGrid(r_gpu, bounds_gpu[2], X_gpu, CUDA.CuVector{Int32}; nthreads = Val(32))
        hgrid_gpu = HashGrid(r_gpu, N, X_gpu, CUDA.CuVector{Int32}; backend = get_backend(X_gpu), nthreads = Val(32))
        @show hgrid_gpu
        #updatecells!(hgrid_gpu, X_gpu)
    end

    F_n = similar(X)
    F_s = similar(X)
    F_p = similar(X)
    F_hg = similar(X)
    F_gpu = similar(X_gpu)
    F_gpu_hg = similar(X_gpu)
    
    if show_results
        printstyled("\nStart benchmarks for N = $(N)\n", color = :blue)
    end 

    forces_multithreaded!(F_p, X, r, grid)
    res_cpu = @benchmark forces_multithreaded!($F_p, $X, $r, $grid)
    df = append_data(df, "CPU (parallel)", N, res_cpu, show_results)

    if hash_cpu
        forces_multithreaded_np!(F_hg, X, r, hgrid)
        res_hash_cpu = @benchmark forces_multithreaded_np!($F_hg, $X, $r, $hgrid)
        df = append_data(df, "CPU (Hash)    ", N, res_hash_cpu, show_results)
    end

    if gpu 
        forces_multithreaded!(F_gpu, X_gpu, r_gpu, grid_gpu)
        res_gpu = @benchmark forces_multithreaded!($F_gpu, $X_gpu, $r_gpu, $grid_gpu)
        df = append_data(df, "GPU           ", N, res_gpu, show_results)
        @assert norm(SVec3f.(F_p) - Vector(F_gpu)) < 0.1 * norm(F_p)
    end

    if hash_gpu 
        #forces_multithreaded_np!(F_gpu_hg, X_gpu, r_gpu, hgrid_gpu)
        res_hash_gpu = @benchmark forces_multithreaded_np!($F_gpu_hg, $X_gpu, $r_gpu, $hgrid_gpu)
        df = append_data(df, "GPU (Hash)    ", N, res_hash_gpu, show_results)
    end

    if celllistmap
        forces_celllistmap!(r, system)
        res_celllistmap = @benchmark forces_celllistmap!($r, $system)
        df = append_data(df, "CellListMap   ", N, res_celllistmap, show_results)
        @assert F_p ≈ system.forces
    end

    if serial
        forces_singlecore!(F_s, X, r, grid)
        res_single = @benchmark forces_singlecore!($F_s, $X, $r, $grid)
        df = append_data(df, "CPU (serial)  ", N, res_single, show_results)
        @assert F_p ≈ F_s 
    end

    if naive && N <= 1_000
        forces_naive!(F_n, X, r)
        res_naive = @benchmark forces_naive!($F_n, $X, $r)
        df = append_data(df, "naive         ", N, res_naive, show_results)
        @assert F_p ≈ F_n
    end 

    
    df[!, :name] = fill(name, length(df.N))

    return df 
end
