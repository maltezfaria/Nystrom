############################ STOKES ############################3
struct Stokes{T}
    μ::T
end
# Single Layer
function (SL::SingleLayerKernel{N,T,Op})(x,y)::T  where {N,T,Op<:Stokes}
    x==y && return zero(T)
    μ = SL.op.μ
    r = x-y
    d = norm(r)
    RRT = r*transpose(r)
    if N==2
        ID = Mat2{Float64}(1,0,0,1)
        γ  = -log(d)
    elseif N==3
        ID = Mat3{Float64}(1,0,0,0,1,0,0,0,1)
        γ  = 1/d 
   end
    return 1/(4π*(N-1)*μ) * (γ*ID + RRT/d^N)
end
SingleLayerKernel{N}(op::Op,args...) where {N,Op<:Stokes} = SingleLayerKernel{N,Mat{N,N,Float64,N*N},Op}(op,args...)

# Double Layer Kernel
function (DL::DoubleLayerKernel{N,T,Op})(x,y,ny)::T where {N,T,Op<:Stokes}
    μ = DL.op.μ
    r = x-y
    d = norm(r)
    if N==2
        x==y && return zero(T)
        return 1/π * vdot(r,ny)/d^4 .* r*transpose(r)

    elseif N==3
        x==y && return zero(T)
        ID = Mat3{Float64}(1,0,0,0,1,0,0,0,1)
        RRT = r*transpose(r) # r ⊗ rᵗ
        return 3/(4π) * vdot(r,ny)/d^5 .* RRT
    end
end
DoubleLayerKernel{N}(op::Op,args...) where {N,Op<:Stokes} = DoubleLayerKernel{N,Mat{N,N,Float64,N*N},Op}(op,args...)


# TODO: code the hypersingular and adjoint double layer operators for Stokes in 2 and 3d
# Adjoint Double Layer Kernel
function (HS::AdjointDoubleLayerKernel{N,T,Op})(x,y,ny)::T where {N,T,Op<:Stokes}
    μ = HS.op.μ
    r = x-y
    d = norm(r)
    if N==2
        x==y && return zero(T)
        return -1/π * vdot(r,ny)/d^4 .* r*transpose(r)
    elseif N==3
        x==y && return zero(T)
        ID = Mat3{Float64}(1,0,0,0,1,0,0,0,1)
        RRT = r*transpose(r) # r ⊗ rᵗ
        return -3/(4π) * vdot(r,ny)/d^5 .* RRT
    end
end
AdjointDoubleLayerKernel{N}(op::Op,args...) where {N,Op<:Stokes} = AdjointDoubleLayerKernel{N,Mat{N,N,Float64,N*N},Op}(op,args...)

# Hypersingular
function (DL::HypersingularKernel{N,T,Op})(x,y,nx,ny)::T where {N,T,Op<:Stokes}
    μ = DL.op.μ
    r = x-y
    d = norm(r)
    if N==2
        x==y && return zero(T)
        return -1/π * vdot(r,ny)/d^4 .* r*transpose(r)
    elseif N==3
        x==y && return zero(T)
        ID = Mat3{Float64}(1,0,0,0,1,0,0,0,1)
        RRT = r*transpose(r) # r ⊗ rᵗ
        return -3/(4π) * vdot(r,ny)/d^5 .* RRT
    end
end
HypersingularKernel{N}(op::Op,args...) where {N,Op<:Stokes} = HypersingularKernel{N,Mat{N,N,Float64,N*N},Op}(op,args...)