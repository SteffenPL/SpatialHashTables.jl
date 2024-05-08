using CUDA

using KernelAbstractions

B = rand(1:2^16, 10^5)
A = cu(B)

B

CUDA.sort!(A)


function countbits!(counts, B, mask)
    fill!(counts, 0)
    @inbounds for b in B
        counts[b & mask] += 1
    end
end

function scan!(offsets, counts)

end

counts = zeros(Int32, 4)

countbits!(counts, B, 2^2 - 1)
counts