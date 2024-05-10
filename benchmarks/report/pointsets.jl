function example_uniform(N)
    r = 1/N^(1/3)
    X = rand(SVector{3, Float64}, N)

    return X, r, (@SVector[0.0,0.0,0.0], @SVector[1.0,1.0,1.0])
end

function example_atomic(N)

end