dist²(z) = sum(x -> x^2, z)
dist(z) = sqrt(dist²(z))


@kernel function interaction_kernel!(compute_fnc, F, @Const(X), grid, cutoff)
    IT = inttype(grid)
    FT = floattype(grid)
    dim = dimension(grid)
    
    i = @index(Global)

    Xi = X[i]
    cutoff² = cutoff * cutoff 

    Fi = zero(eltype(F))
    for j in neighbours(grid, Xi, cutoff)
        Xj = X[j]
        Xij = Xi - Xj
        d² = dist²(Xij)

        if i != j && d² <= cutoff²  
            Fi = compute_fnc(Fi, i, j, Xi, Xj, Xij, d²)
        end
    end
    F[i] = Fi

end

function compute_interactions!(interaction_fnc, F, X, grid, cutoff)

    compute_fnc(Fi, i, j, Xi, Xj, Xij, d²) = Fi + interaction_fnc(i, j, Xi, Xj, Xij, d²)

    kernel = interaction_kernel!(grid.backend, unval(grid.nthreads))
    kernel(compute_fnc, F, X, grid, cutoff, ndrange = length(X))
    KernelAbstractions.synchronize(grid.backend)

    return F
end



