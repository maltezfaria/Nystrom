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
        R  = error_interior_green_identity(SL,DL,γ₀B,γ₁B)
    elseif kernel_type(iop) isa Union{AdjointDoubleLayer,HyperSingular}
        ADL = IntegralOperator{T}(AdjointDoubleLayerKernel(op),X,Y) |> Matrix
        H   = IntegralOperator{T}(HyperSingularKernel(op),X,Y) |> Matrix
        R   = error_interior_derivative_green_identity(ADL,H,γ₀B,γ₁B)
    end

    # compute the interpolation matrix
    a,b       = combined_field_coefficients(kernel)
    near_list = nearest_point_list(X.quadrature,Y.quadrature,tol=1)
    for i in 1:m # loop over rows
        for (idx_el,idx_node) in near_list[i]
            yels = getelements(Y)[idx_el]
            nnear = length(yels)
            L                = Matrix{T}(undef,2*nnear,nbasis)
            D                = [diagm(fill(a*one(T),nnear));diagm(fill(b*one(T),nnear))]
            L[1:nnear,:]     = γ₀B[yels,:]
            L[nnear+1:end,:] = γ₁B[yels,:]
            invL = invert_green_matrix(L)
            if getnodes(X)[i] == getnodes(Y)[idx_node]
                w[i]        = (R[i:i,:]*invL)*D |> vec
            else
                w[i]        = ((R[i:i,:]+γ₀B[i:i,:]/2)*invL)*D |> vec
            end
            idx_near[i] = copy(yels)
        #NOTE: making a copy is important here or you get issues when
        #mutating the GreesCorrection since mutating the non-zero indices of
        #one row would inadvertently affect another.
        end
    end
    return GreensCorrection{T}(kernel,X,Y,w,idx_near)
end

function invert_green_matrix(L::Matrix{Mat{M,N,T,S}}) where {M,N,T,S}
    Lfull = Matrix(L)
    Linv = invert_green_matrix(Lfull)
    return matrix_to_tensor(Mat{M,N,eltype(Linv),S},Linv)
end

function invert_green_matrix(L::AbstractMatrix{T},S=Double64) where {T<:Number}
    if T==Float64
        L2 = S.(L)
    elseif T==ComplexF64
        L2 = Complex{S}.(L)
    end
    Linv = @suppress pinv(L2,rtol=eps(S))
    return Linv
end

function GreensCorrection(iop::IntegralOperator,xs)
    # construct greens "basis" from source locations xs
    op        = iop.kernel.op
    basis     = [y->SingleLayerKernel(op)(x,y) for x in xs]
    γ₁_basis  = [(y,ny)->transpose(DoubleLayerKernel(op)(x,y,ny)) for x in xs]
    GreensCorrection(iop,basis,γ₁_basis)
end

function GreensCorrection(iop::IntegralOperator)
    nquad  = mapreduce(x->length(x),max,getelements(iop.Y))
    nbasis = 3*nquad + 2
    # construct source basis
    xs     = source_gen(iop.Y,nbasis)
    GreensCorrection(iop,xs)
end

error_green_formula(SL,DL,γ₀u,γ₁u,σ)                      = σ*γ₀u + SL*γ₁u  - DL*γ₀u
error_derivative_green_formula(ADL,H,γ₀u,γ₁u,σ)           = σ*γ₁u + ADL*γ₁u - H*γ₀u
error_interior_green_identity(SL,DL,γ₀u,γ₁u)              = error_green_formula(SL,DL,γ₀u,γ₁u,-1/2)
error_interior_derivative_green_identity(ADL,H,γ₀u,γ₁u)   = error_derivative_green_formula(ADL,H,γ₀u,γ₁u,-1/2)
error_exterior_green_identity(SL,DL,γ₀u,γ₁u)              = error_green_formula(SL,DL,γ₀u,γ₁u,1/2)
error_exterior_derivative_green_identity(ADL,H,γ₀u,γ₁u)  = error_derivative_green_formula(ADL,H,γ₀u,γ₁u,+1/2)

"""
    near_interaction_list(X,Y;[tol],)

Return a vector of `length(X)` elements, where each element contains the index
in `Y` of nodes for which `norm(x-y)<tol`. For points which belong to the same
element in `Y`, only the closest one is returned.
"""
function near_interaction_list(X,Y; tol=0)
    n = length(X)
    list = [Vector{Tuple{Int,Int}}() for _ = 1:n] # (idxel, idxnode) of near points for each x ∈ X
    xnodes = getnodes(X)
    ynodes = getnodes(Y)
    yels   = getelements(Y)
    for i in 1:n
        x = xnodes[i]
        for nel in 1:length(yels)
            idxnodes = yels[nel]
            d = map(j -> norm(x .- ynodes[j]),idxnodes)
            dmin,idx_min = findmin(d)
            if dmin ≤ tol
                push!(list[i],(nel,idxnodes[idx_min]))
            end
        end
    end
    return list
end

"""
    nearest_point_list(X,Y;[tol])

Return a vector of `length(X)` elements, where each element contains the index
in `Y` of nodes for which `norm(x-y)<tol`. For points which belong to the same
element in `Y`, only the closest one is returned.
"""
function nearest_point_list(X,Y; tol=0)
    n,m  = length(X),length(Y)
    list = [Vector{Tuple{Int,Int}}() for _ = 1:n] # (idxel, idxnode) of near points for each x ∈ X
    xnodes = getnodes(X)
    ynodes = getnodes(Y)
    in2e   = idx_nodes_to_elements(Y)
    for i in 1:n
        x = xnodes[i]
        d = map(j -> norm(x .- ynodes[j]),1:m)
        dmin,idx_min = findmin(d)
        if dmin ≤ tol
            push!(list[i],(in2e[idx_min][1],idx_min))
        end
    end
    return list
end

"""
    near_interaction_list_bodies(X,Y;[tol],)

Return a vector of `length(X)` elements, where each element contains the index
in `Y` of nodes for which `norm(x-y)<tol`. For points which belong to the same
body in `Y`, only the closest one is returned.
"""
function near_interaction_list_bodies(X,Y; tol=0)
    n         = length(X)
    list      = [Vector{Tuple{Int,Int}}() for _ = 1:n] # (idxel, idxnode) of near points for each x ∈ X
    xnodes    = getnodes(X)
    ynodes    = getnodes(Y)
    bdy2nodes      = idx_bodies_to_nodes(Y)
    nodes2elements = idx_nodes_to_elements(Y)
    for i in 1:n
        x = xnodes[i]
        for k in 1:length(getbodies(Y))
            idxnodes = bdy2nodes[k]
            d = map(j -> norm(x .- ynodes[j]),idxnodes)
            dmin,idx_min = findmin(d)
            if dmin ≤ tol
                imin = idxnodes[idx_min]
                push!(list[i],(nodes2elements[imin][1],imin))
            end
        end
    end
    return list
end
