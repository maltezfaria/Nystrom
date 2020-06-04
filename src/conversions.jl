# convert an Array of SMatrices to a standard array to use the linear algebra algorithms such as pinv
function Base.Matrix(A::Array{Mat{M,N,T,L},2})  where {M,N,T,L}
    sblock = (M,N)
    ss = size(A).*sblock
    Afull = Matrix{T}(undef,ss)
    for i=1:ss[1], j=1:ss[2]
        bi, ind_i = divrem(i-1,sblock[1]) .+ (1,1)
        bj, ind_j = divrem(j-1,sblock[2]) .+ (1,1)
        Afull[i,j] = A[bi,bj][ind_i,ind_j]
    end
    return Afull
end

function matrix_to_blockmatrix(T,A::Matrix)
    sblock = size(T)
    nblock = div.(size(A),sblock)
    Ablock = Matrix{T}(undef,nblock)
    for i in 1:nblock[1]
        istart = (i-1)*sblock[1] + 1
        iend = i*sblock[1]
        for j in 1:nblock[2]
            jstart = (j-1)*sblock[2] + 1
            jend   = j*sblock[2]
            Ablock[i,j] = A[istart:iend,jstart:jend]
        end
    end
    return Ablock
end

# function fill_matrix(IOp::IntegralOperator{N,T})  where {N,T}
#     npts = length(IOp.quad.weights)
#     A = [IOp[i,j] for i=1:npts, j=1:npts]
# end

# function Base.vec(Vector{Vec{N,T}}) where {N,T}
#     reinterpret
# end
