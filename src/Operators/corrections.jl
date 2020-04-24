"""
    GreensCorrection{T,S} <: AbstractMatrix{T}

An `AbstractMatrix` representing a correction to a singular boundary integral
operator using density interpolation method with Greens functions [^1].

The underlying representation is *sparse* since only near-field interactions need to be taken into account.
This structure is typically added to the dense part of an integral operator as a correction.

[^1] TODO: CITE OUR PAPER
"""
struct GreensCorrection{T,K,S,V} <: AbstractMatrix{T}
    kernel::K
    X::S
    Y::V
    weights::Vector{Vector{T}}
    idx_near::Vector{Vector{Int}}
end
GreensCorrection{T}(k,X,Y,args...) where {T} = GreensCorrection{T,typeof(k),typeof(X),typeof(Y)}(k,X,Y,args...)

Base.size(c::GreensCorrection) = length(c.X),length(c.Y)

function Base.getindex(c::GreensCorrection,i::Int,j::Int)
    T = eltype(c)
    idxs = c.idx_near[i]
    for (n,jnear) in enumerate(idxs)
        if jnear == j
            return c.weights[i][n]
        end
    end
    return zero(T)
end

function Base.setindex!(c::GreensCorrection,v,i::Int,j::Int)
    T = eltype(c)
    idxs = c.idx_near[i]
    for (n,jnear) in enumerate(idxs)
        if jnear == j
            c.weights[i][n] = v
            return c
        end
    end
    push!(idxs,j)
    push!(c.weights[i],v)
    return c
end

function GreensCorrection(iop::IntegralOperator{T,K}, basis, γ₁_basis) where {T,K}
    σ = 0.5
    kernel,X,Y  = iop.kernel, iop.X, iop.Y
    op          = kernel.op
    m,n         = length(X),length(Y)
    w           = [Vector{T}() for _ in 1:m]
    idx_near    = [Vector{Int}() for _ in 1:m]

    nbasis = length(basis)

    # compute matrix of basis evaluated on Y
    γ₀B = Matrix{T}(undef,length(Y),nbasis)
    γ₁B = Matrix{T}(undef,length(Y),nbasis)
    for n in 1:nbasis
        γ₀B[:,n] = γ₀(basis[n],Y)
        γ₁B[:,n] = γ₁(γ₁_basis[n],Y)
    end

    # integrate the basis
    if kernel_type(iop) isa Union{SingleLayer,DoubleLayer}
        SL = IntegralOperator{T}(SingleLayerKernel(op),X,Y) |> Matrix
        DL = IntegralOperator{T}(DoubleLayerKernel(op),X,Y) |> Matrix
        R  = error_interior_greens_identity(SL,DL,γ₀B,γ₁B)
    elseif kernel_type(iop) isa Union{AdjointDoubleLayer,HyperSingular}
        R  = error_interior_derivative_greens_identity(op,γ₀B,γ₁B)
    end

    # compute the interpolation matrix
    a,b       = combined_field_coefficients(kernel)
    near_list = near_interaction_list(X,Y)
    _prune_interaction_list!(near_list,X,Y)

    for i in 1:m # loop over rows
        for (idx_el,idx_node) in near_list[i]
            yels = getelements(Y)[idx_el]
            nnear = length(yels)
            L                = Matrix{T}(undef,2*nnear,nbasis)
            D                = [diagm(fill(a,nnear));diagm(fill(b,nnear))]
            L[1:nnear,:]     = γ₀B[yels,:]
            L[nnear+1:end,:] = γ₁B[yels,:]
            w[i] = (R[idx_node:idx_node,:]*pinv(L))*D |> vec
            idx_near[i] = copy(yels) 
            #NOTE: making a copy is important here or you get issues when mutating the GreesCorrection since mutating the non-zero
            #indices of one row would inadvertently affect another.
        end
    end
    return GreensCorrection{T}(kernel,X,Y,w,idx_near)
end

error_interior_greens_identity(SL,DL,γ₀u,γ₁u)            = -γ₀u/2 + SL*γ₁u - DL*γ₀u
error_interior_derivative_greens_identity(ADL,H,γ₀u,γ₁u) = -γ₁u/2 + ADL*γ₁u - H*γ₀u
error_exterior_greens_identity(SL,DL,γ₀u,γ₁u)            = γ₀u/2 + SL*γ₁u - DL*γ₀u
error_exterior_derivative_greens_identity(SL,DL,γ₀u,γ₁u) = γ₁u/2 + ADL*γ₁u - H*γ₀u

# function OldGreensCorrection(iop,)
#     idx_el2n    = getelements(Y)
#     nel         = length(idx_el2n)
#     qsize = length(IOp.Y |> quadrature_type)
#     @show nel, ndof, qsize
#     SL_kernel   = SingleLayerKernel(op)
#     DL_kernel   = DoubleLayerKernel(op)
#     γ₀B         = [transpose(SL_kernel(x,y)) for y in quad.nodes,  x in xₛ]
#     γ₁B         = [transpose(DL_kernel(x,y,ny)) for (y,ny) in zip(quad.nodes,quad.normals), x in xₛ ]

#     if kernel_type(IOp) isa Union{SingleLayer,DoubleLayer}
#         a,b = 1,0
#         R  = Op2*γ₀B - Op1*γ₁B + σ*γ₀B
#     elseif kernel_type(IOp) isa Union{AdjointDoubleLayer,HyperSingular}
#         a,b = 0,-1
#         R  = Op2*γ₀B - Op1*γ₁B + σ*γ₁B
#     end
#     L      = Vector{Matrix{T}}(undef,nel)
#     for iel in 1:nel
#         idx       = elements[iel]
#         k         = length(idx)
#         D         = [diagm(fill(a,k));diagm(fill(b,k))]
#         A         = Matrix(γ₀B[idx,:])
#         B         = Matrix(γ₁B[idx,:])
#         L[iel]    = pinv([A;B])*D
#     end
#     return GreensCorrection(γ₀B,γ₁B,L,R,deepcopy(elements))
# end

# function (corr::GreensCorrection)(x::AbstractVector)
#     TS = promote_type(eltype(corr),eltype(x))
#     idxs = corr.idxs
#     R    = corr.R
#     out = zero(TS,x)
#     for iel in 1:length(idxs)
#         tmp = L[iel]*x[idxs[iel]]
#         for i in iel
#             out[i] += R[i:i,:]*tmp
#         end
#     end
#     return out
# end

# function Base.getindex(corr::GreensCorrection{N,T},i::Int,j::Int) where {N,T}
#     if abs(i-j) >= corr.quad.nodes_per_element
#         return zero(T)
#     else
#         δj = [i==j for i=1:leg]
#     end
# end
