"""
    GreensCorrection{T,S} <: AbstractMatrix{T}

An `AbstractMatrix` representing a correction to a singular boundary integral
operator using density interpolation method with Greens functions [^1].

The underlying representation is *sparse* since only near-field interactions need to be taken into account.
This structure is typically added to the dense part of an integral operator as a correction.

[^1] TODO: CITE OUR PAPER
"""
struct GreensCorrection{T,S,U} <: AbstractMatrix{T}
    iop::S
    R::Matrix{T}
    L::U
    idxel_near::Vector{Int}
end

Base.size(c::GreensCorrection) = size(c.iop)

function Base.getindex(c::GreensCorrection,i::Int,j::Int)
    iop = c.iop
    T = eltype(c)
    a,b       = combined_field_coefficients(iop.kernel)
    idx_el    = c.idxel_near[i]
    idx_el < 0 && (return zero(T))
    idx_nodes = getelements(iop.Y)[idx_el]
    for (n,jnear) in enumerate(idx_nodes)
        if jnear == j
            L       = c.L[idx_el]
            ninterp = length(idx_nodes)
            D = [diagm(fill(a*one(T),ninterp));diagm(fill(b*one(T),ninterp))]
            tmp = c.R[i:i,:]*invert_green_matrix(L)
            w   = @views a*tmp[1:ninterp] + b*tmp[ninterp+1:end]
            w = tmp*D
            return w[n]
        end
    end
    return zero(T)
end

function GreensCorrection(iop::IntegralOperator{T,K},Op1,Op2,basis,γ₁_basis,σ) where {T,K}
    @assert σ == 0 || σ == -0.5
    kernel,X,Y  = iop.kernel, iop.X, iop.Y
    op          = kernel.op
    m,n         = length(X),length(Y)
    L           = Vector{Matrix{T}}(undef,length(getelements(Y)))

    nbasis      = length(basis)

    # compute matrix of basis evaluated on Y
    γ₀B = Matrix{T}(undef,length(Y),nbasis)
    γ₁B = Matrix{T}(undef,length(Y),nbasis)
    for k in 1:nbasis
        γ₀B[:,k] = γ₀(basis[k],Y)
        γ₁B[:,k] = γ₁(γ₁_basis[k],Y)
    end

    # integrate the basis over Y
    R  = Op1*γ₁B - Op2*γ₀B
    if kernel_type(iop) isa Union{SingleLayer,DoubleLayer}
        if σ == -0.5
            R  += σ*γ₀B
        end
    elseif kernel_type(iop) isa Union{AdjointDoubleLayer,HyperSingular}
        if σ == -0.5
            R  += σ*γ₁B
        end
    end

    # compute the interpolation matrix
    idxel_near = nearest_element_list(X,Y,tol=1)
    for i in 1:m # loop over rows
        idx_el = idxel_near[i]
        idx_el < 0 && continue
        idx_nodes  = getelements(Y)[idx_el]
        ninterp    = length(idx_nodes)
        M          = Matrix{T}(undef,2*ninterp,nbasis)
        M[1:ninterp,:]     = γ₀B[idx_nodes,:]
        M[ninterp+1:end,:] = γ₁B[idx_nodes,:]
        L[idx_el]          = M
    end
    return GreensCorrection(iop,R,L,idxel_near)
end

function GreensCorrection(iop::IntegralOperator,Op1,Op2,xs::Vector{<:Point})
    # construct greens "basis" from source locations xs
    op        = iop.kernel.op
    basis     = [y->SingleLayerKernel(op)(x,y) for x in xs]
    γ₁_basis  = [(y,ny)->transpose(DoubleLayerKernel(op)(x,y,ny)) for x in xs]
    σ = iop.X === iop.Y ? -0.5 : 0
    GreensCorrection(iop,Op1,Op2,basis,γ₁_basis,σ)
end

function GreensCorrection(iop::IntegralOperator,Op1,Op2)
    xs = source_gen(iop)
    GreensCorrection(iop,Op1,Op2,xs)
end

function GreensCorrection(iop::IntegralOperator,compress=Matrix)
    X,Y,op = iop.X, iop.Y, iop.kernel.op
    T = eltype(iop)
    # construct integral operators required for correction
    if kernel_type(iop) isa Union{SingleLayer,DoubleLayer}
        Op1 = IntegralOperator{T}(SingleLayerKernel(op),X,Y) |> compress
        Op2 = IntegralOperator{T}(DoubleLayerKernel(op),X,Y) |> compress
    elseif kernel_type(iop) isa Union{AdjointDoubleLayer,HyperSingular}
        Op1 = IntegralOperator{T}(AdjointDoubleLayerKernel(op),X,Y) |> compress
        Op2 = IntegralOperator{T}(HyperSingularKernel(op),X,Y) |> compress
    end
    GreensCorrection(iop,Op1,Op2)
end

################################################################################
################################################################################
################################################################################
# FIXME:this function needs to be cleaned up and optimized for perf
function precompute_weights_qr(c::GreensCorrection{T}) where {T<:Number}
    w = [Vector{T}() for _ in 1:size(c,1)]
    iop  = c.iop
    a,b  = combined_field_coefficients(iop.kernel)
    X,Y  = iop.X, iop.Y
    QRType = Base.promote_op(qr,Matrix{T})
    LQR  = Vector{QRType}(undef,length(c.L))
    for i in 1:size(c,1)
        idx_el    = c.idxel_near[i]
        idx_el < 0 && continue
        idx_nodes = getelements(iop.Y)[idx_el]
        ninterp = length(idx_nodes)
        if !isassigned(LQR,idx_el)
            LQR[idx_el] = qr(c.L[idx_el])
        end
        F       = LQR[idx_el]
        tmp     = (c.R[i:i,:]*pinv(F.R))*adjoint(F.Q)
        w[i]     = @views a*tmp[1:ninterp] + b*tmp[ninterp+1:end]
    end
    return w
end

# Tensor case. We need to convert the *matrix of matrices* structures to flat
# matrices for doing qr, then convert back. Not very efficient... but may be not
# very important?
function precompute_weights_qr(c::GreensCorrection{T}) where {T<:Mat}
    w   = [Vector{T}() for _ in 1:size(c,1)]
    iop = c.iop
    a,b = combined_field_coefficients(iop.kernel)
    X,Y = iop.X, iop.Y
    QRType = Base.promote_op(qr,Matrix{eltype(T)})
    LQR    = Vector{QRType}(undef,length(c.L))
    for i in 1:size(c,1)
        idx_el    = c.idxel_near[i]
        idx_el < 0 && continue
        idx_nodes = getelements(iop.Y)[idx_el]
        ninterp = length(idx_nodes)
        if !isassigned(LQR,idx_el)
            LQR[idx_el] = qr(Matrix(c.L[idx_el]))
        end
        F    = LQR[idx_el]
        tmp  = (Matrix(c.R[i:i,:])*pinv(F.R))*adjoint(F.Q)
        tmp2 = matrix_to_blockmatrix(T,tmp)
        w[i] = @views a*tmp2[1:ninterp] + b*tmp2[ninterp+1:end]
    end
    return w
end

function precompute_weights_pinv(c::GreensCorrection,S=Float64)
    T = eltype(c)
    w = [Vector{T}() for _ in 1:size(c,1)]
    iop  = c.iop
    a,b  = combined_field_coefficients(iop.kernel)
    X,Y   = iop.X, iop.Y
    Linv  = [invert_green_matrix(Matrix(L),S) for L in c.L]
    for i in 1:size(c,1)
        idx_el    = c.idxel_near[i]
        idx_el < 0 && continue
        idx_nodes = getelements(iop.Y)[idx_el]
        ninterp = length(idx_nodes)
        tmp     = Matrix(c.R[i:i,:])*Linv[idx_el]
        if T <: Mat
            tmp  = matrix_to_blockmatrix(T,tmp)
        end
        w[i]     = @views a*tmp[1:ninterp] + b*tmp[ninterp+1:end]
    end
    return w
end

function invert_green_matrix(L::AbstractMatrix{T},S=Float64) where {T<:Number}
    if T==Float64
        L2 = S.(L)
        Linv = @suppress pinv(L2,rtol=eps(S))
    elseif T==ComplexF64
        L2   = Complex{S}.(L)
        Linv = @suppress pinv(L2,rtol=eps(S))
    end
    er = norm(L*Linv - I,Inf)
    if er > 1e-12
        if S===Float64
            Linv = invert_green_matrix(L,Double64)
        else
            @debug er
        end
    end
    return Linv
end

function Base.Matrix(c::GreensCorrection)
    iop = c.iop
    a,b   = combined_field_coefficients(iop.kernel)
    M     = zero(c)
    w     = precompute_weights_qr(c)
    for i in 1:size(c,1)
        idx_el    = c.idxel_near[i]
        idx_el < 0 && continue
        idx_nodes = getelements(iop.Y)[idx_el]
        M[i,idx_nodes] = w[i]
    end
    return M
end

function SparseArrays.sparse(c::GreensCorrection)
    m,n = size(c)
    iop = c.iop
    T   = eltype(c)
    a,b = combined_field_coefficients(iop.kernel)
    I = Int[]
    J = Int[]
    V = T[]
    w = precompute_weights_qr(c)
    for i in 1:size(c,1)
        idx_el    = c.idxel_near[i]
        idx_el < 0 && continue
        idx_nodes = getelements(iop.Y)[idx_el]
        append!(I,fill(i,length(idx_nodes)))
        append!(J,idx_nodes)
        append!(V,w[i])
    end
    return sparse(I,J,V,m,n)
end

function LinearAlgebra.axpy!(a,X::GreensCorrection,Y::Matrix)
    iop = X.iop
    c1,c2 = combined_field_coefficients(iop.kernel)
    w     = precompute_weights_qr(X)
    for i in 1:size(X,1)
        idx_el    = X.idxel_near[i]
        idx_el < 0 && continue
        idx_nodes = getelements(iop.Y)[idx_el]
            axpy!(a,w[i],view(Y,i,idx_nodes))
    end
    return Y
end

Base.:+(X::GreensCorrection,Y::Matrix) = axpy!(true,X,copy(Y))
Base.:+(X::Matrix,Y::GreensCorrection) = Y+X

error_green_formula(SL,DL,γ₀u,γ₁u,u,σ)                      = σ*u + SL*γ₁u  - DL*γ₀u
error_derivative_green_formula(ADL,H,γ₀u,γ₁u,un,σ)          = σ*un + ADL*γ₁u - H*γ₀u
error_interior_green_identity(SL,DL,γ₀u,γ₁u)                = error_green_formula(SL,DL,γ₀u,γ₁u,γ₀u,-1/2)
error_interior_derivative_green_identity(ADL,H,γ₀u,γ₁u)     = error_derivative_green_formula(ADL,H,γ₀u,γ₁u,γ₁u,-1/2)
error_exterior_green_identity(SL,DL,γ₀u,γ₁u)                = error_green_formula(SL,DL,γ₀u,γ₁u,γ₀u,1/2)
error_exterior_derivative_green_identity(ADL,H,γ₀u,γ₁u)     = error_derivative_green_formula(ADL,H,γ₀u,γ₁u,γ₁u,1/2)

"""
    nearest_element_list(X,Y;[tol])

Return a vector of integers, where the `i` entry of the vector gives the index of the nearest element in `Y` to the *ith*-node.

An optional keywork argument `tol` can be passed so that only elements which are closer than `tol` are considered. If a node `x ∈ X` with index `i` has no element in `Y` closer than `tol`, the value -1 is stored indicating such a case.
"""
function nearest_element_list(X,Y; tol=0)
    n,m  = length(X),length(Y)
    list = fill(-1,n) # idxel of nearest element for each x ∈ X. -1 means there is not element in `Y` closer than `tol`
    xnodes = getnodes(X)
    ynodes = getnodes(Y)
    in2e   = idx_nodes_to_elements(Y)
    # special case (which arises often for integral operators) where X==Y.
    # No distance computation is needed then
    if X == Y
        for i=1:n
            @assert length(in2e[i]) == 1
            list[i] = in2e[i] |>  first
        end
    end
    # compute the nearest element for each node FIXME: this is n^2 complexity,
    # should do an n complexity by using e.g. a cluster tree
    for i in 1:n
        x = xnodes[i]
        d = map(j -> norm(x .- ynodes[j]),1:m)
        dmin,idx_min = findmin(d)
        if dmin ≤ tol
            @assert length(in2e[idx_min]) == 1
            list[i] = in2e[idx_min] |>  first
        end
    end
    return list
end

"""
    idx_nodes_to_elements(q::Quadrature)

For each node in `q`, return the indices of the elements to which it belongs.

Depending on the quadrature type, more efficient methods can be defined and overloaded if needed.
"""
function idx_nodes_to_elements(q)
    list = [Int[] for _ in 1:length(q)]
    for n in 1:length(getelements(q))
        for i in getelements(q)[n]
            push!(list[i],n)
        end
    end
    return list
end
