############################ Helmholtz ############################
struct Helmholtz{N,T}
    k::T
end
Helmholtz(;k,dim=3) = Laplace{dim,typeof(k)}(k)
# Single Layer
function (SL::SingleLayerKernel{N,T,Op})(x,y)::T  where {N,T,Op<:Helmholtz}
    x==y && return zero(T)
    k = SL.op.k
    r = x-y
    d = sqrt(sum(r.^2))
    # d = norm(x-y)
    if N==2
        return im/4 * hankelh1(0,k*d)
    elseif N==3
        return 1/(4π)/d * exp(im*k*d)
    end
end
SingleLayerKernel{N}(op::Op,args...) where {N,Op<:Helmholtz} = SingleLayerKernel{N,ComplexF64,Op}(op,args...)

# Double Layer Kernel
function (DL::DoubleLayerKernel{N,T,Op})(x,y,ny)::T where {N,T,Op<:Helmholtz}
    x==y && return 0
    k = DL.op.k
    r = x-y
    d = sqrt(sum(r.^2))
    # d = norm(x-y)
    if N==2
        return im*k/4/d * hankelh1(1,k*d) .* dot(r,ny)
    elseif N==3
        return 1/(4π)/d^2 * exp(im*k*d) * ( -im*k + 1/d ) * vdot(r,ny)
    end
end
DoubleLayerKernel{N}(op::Op,args...) where {N,Op<:Helmholtz} = DoubleLayerKernel{N,ComplexF64,Op}(op,args...)

# Adjoint double Layer Kernel
function (ADL::AdjointDoubleLayerKernel{N,T,Op})(x,y,nx)::T where {N,T,Op<:Helmholtz}
    x==y && return 0
    k = ADL.op.k
    r = x-y
    d = norm(r)
    if N==2
        return -im*k/4/d * hankelh1(1,k*d) .* vdot(r,nx)
    elseif N==3
        return -1/(4π)/d^2 * exp(im*k*d) * ( -im*k + 1/d ) * vdot(r,nx)
    end
end
AdjointDoubleLayerKernel{N}(op::Op,args...) where {N,Op<:Helmholtz} = AdjointDoubleLayerKernel{N,ComplexF64,Op}(op,args...)

# Hypersingular kernel
function (HS::HypersingularKernel{N,T,Op})(x,y,nx,ny)::T where {N,T,Op<:Helmholtz}
    x==y && return 0
    k = HS.op.k
    r = x-y
    d = norm(x-y)
    if N==2
        ID = Mat2{Float64}(1,0,0,1)
        RRT = r*transpose(r) # r ⊗ rᵗ
        # TODO: rewrite the operation below in a more clear/efficient way
        return transpose(nx)*((-im*k^2/4/d^2*hankelh1(2,k*d).*RRT + im*k/4/d*hankelh1(1,k*d).*ID)*ny)
    elseif N==3
        ID = Mat3{Float64}(1,0,0,0,1,0,0,0,1)
        RRT = r*transpose(r) # r ⊗ rᵗ
        term1 = 1/(4π)/d^2 * exp(im*k*d) * ( -im*k + 1/d ) * ID
        term2 = RRT/d * exp(im*k*d)/(4*π*d^4) * (3*(d*im*k-1) + d^2*k^2)
        return  transpose(nx)*(term1 + term2) * ny
    end
end
HypersingularKernel{N}(op::Op,args...) where {N,Op<:Helmholtz} = HypersingularKernel{N,ComplexF64,Op}(op,args...)
