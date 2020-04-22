struct IOpCorrection{N,T,T2,T3,Tmesh} <: AbstractMatrix{T}
    γ₀B::Array{T,2}
    γ₁B::Array{T,2}
    L::Vector{T2}
    R::Array{T3,2} #integrated contribution of correction
    quad::Quadrature{N,Tmesh}
end

function IOpCorrection(IOp::IntegralOperator{N,T}, Op1, Op2, xₛ, σ = 0.5) where {N,T}
    op          = IOp.kernel.op
    quad        = IOp.quad
    SL_kernel   = SingleLayerKernel{N}(op)
    DL_kernel   = DoubleLayerKernel{N}(op)
    npts        = length(quad.nodes)
    γ₀B         = [transpose(SL_kernel(x,y)) for y in quad.nodes,  x in xₛ]
    γ₁B         = [transpose(DL_kernel(x,y,ny)) for (y,ny) in zip(quad.nodes,quad.normals), x in xₛ ]

    qsize = quad.nodes_per_element
    if T <: Number
        s1 =1; s2 = 1;
    else
        s1, s2 = size(T)
    end
    nrow  = 2*qsize*s1
    ncol  = length(xₛ)*s2
    nmat = div(npts,qsize)
    # L    = Vector{Mat{ncol,nrow, eltype(T),nrow*ncol}}(undef,nmat)
    L    = Vector{Mat{nrow,ncol, eltype(T),nrow*ncol}}(undef,nmat)
    for i in 1:qsize:npts
        idxs      = i:i+qsize-1
        A         = Matrix(γ₀B[idxs,:])
        B         = Matrix(γ₁B[idxs,:])
        # L[div(i-1,qsize)+1]      = pinv([A;B])
        L[div(i-1,qsize)+1]      = [A;B]
    end

    if isa(IOp, Union{SingleLayerOperator,DoubleLayerOperator})
        R  = Op2*γ₀B - Op1*γ₁B + σ*γ₀B
    else
        R  = Op2*γ₀B - Op1*γ₁B + σ*γ₁B
    end
    return IOpCorrection(γ₀B,γ₁B,L,R,quad)
end

function (corr::IOpCorrection)(φ::AbstractVector{T},α,β) where T
    quad = corr.quad
    qsize = quad.nodes_per_element
    out = zero(φ)
    for i in 1:qsize:length(quad.nodes)
        idxs      = i:i+qsize-1
        rhs       = [α*φ[idxs]; β*φ[idxs]]
        rhs       = reinterpret(eltype(T),rhs)
        ind       = div(i-1,qsize) + 1
        # c         = corr.L[ind]*rhs
        c         = Matrix(corr.L[ind])\rhs
        if !(T <: Number)
            c         = reinterpret(T,reshape(c,size(T,1),:))
        end
        for j=0:qsize-1
            idx = i+j
            for n=1:length(c)
                out[idx] += corr.R[idx,n]*c[n]
            end
        end
    end
    return out
end

# function Base.getindex(corr::IOpCorrection{N,T},i::Int,j::Int) where {N,T}
#     if abs(i-j) >= corr.quad.nodes_per_element
#         return zero(T)
#     else
#         δj = [i==j for i=1:leg]
#     end
# end

