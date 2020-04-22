############################ ELASTOSTATIC ############################3
struct Elastostatic{T}
    μ::T
    λ::T
end
# Single Layer
function (SL::SingleLayerKernel{N,T,Op})(x,y)::T  where {N,T,Op<:Elastostatic}
    x==y && return zero(T)
    μ = SL.op.μ
    λ = SL.op.λ
    ν = λ/(2*(μ+λ))
    r = x-y
    d = norm(r)
    RRT = r*transpose(r) # r ⊗ rᵗ
    if N==2
        ID = Mat2{Float64}(1,0,0,1)
        return 1/(8π*μ*(1-ν))*(-(3-4*ν)*log(d)*ID + RRT/d^2)
    elseif N==3
        ID = Mat3{Float64}(1,0,0,0,1,0,0,0,1)
        return 1/(16π*μ*(1-ν)*d)*((3-4*ν)*ID + RRT/d^2)
    end
end
SingleLayerKernel{N}(op::Op,args...) where {N,Op<:Elastostatic} = SingleLayerKernel{N,Mat{N,N,Float64,N*N},Op}(op,args...)

# Double Layer Kernel
function (DL::DoubleLayerKernel{N,T,Op})(x,y,ny)::T where {N,T,Op<:Elastostatic}
    x==y && return zero(T)
    μ = DL.op.μ
    λ = DL.op.λ
    ν = λ/(2*(μ+λ))
    r = x-y
    d = norm(r)
    RRT = r*transpose(r) # r ⊗ rᵗ
    drdn = vdot(r,ny)/d
    if N==2
        ID = Mat2{Float64}(1,0,0,1)
        return -1/(4π*(1-ν)*d)*(drdn*((1-2ν)*ID + 2*RRT/d^2) - (1-2ν)/d*(r*transpose(ny) - ny*transpose(r)))
    elseif N==3
        ID = Mat3{Float64}(1,0,0,0,1,0,0,0,1)
        return -1/(8π*(1-ν)*d^2)*(drdn * ((1-2*ν)*ID + 3*RRT/d^2) - (1-2*ν)/d*(r*transpose(ny) - ny*transpose(r)))
    end
end
DoubleLayerKernel{N}(op::Op,args...) where {N,Op<:Elastostatic} = DoubleLayerKernel{N,Mat{N,N,Float64,N*N},Op}(op,args...)

# Adjoint Double Layer Kernel
function (ADL::AdjointDoubleLayerKernel{N,T,Op})(x,y,nx)::T where {N,T,Op<:Elastostatic}
    x==y && return zero(T)
    μ = ADL.op.μ
    λ = ADL.op.λ
    ν = λ/(2*(μ+λ))
    r = x-y
    d = norm(r)
    RRT = r*transpose(r) # r ⊗ rᵗ
    if N==2
        ID = Mat2{Float64}(1,0,0,1)
        return 1/(4π*(1-ν)*d)*((1-2ν)/d*(-nx*transpose(r) + vdot(r,nx)*ID + r*transpose(nx)) + 2/d^3*vdot(r,nx)*RRT)
    elseif N==3
        ID = Mat3{Float64}(1,0,0,0,1,0,0,0,1)
        return 1/(8π*(1-ν)*d^2)*((1-2ν)/d*(-nx*transpose(r) + vdot(r,nx)*ID + r*transpose(nx)) + 3/d^3*vdot(r,nx)*RRT)
    end
end
AdjointDoubleLayerKernel{N}(op::Op,args...) where {N,Op<:Elastostatic} = AdjointDoubleLayerKernel{N,Mat{N,N,Float64,N*N},Op}(op,args...)

# Hypersingular kernel
function (HS::HypersingularKernel{N,T,Op})(x,y,nx,ny)::T where {N,T,Op<:Elastostatic}
    x==y && return zero(T)
    μ = HS.op.μ
    λ = HS.op.λ
    ν = λ/(2*(μ+λ))
    r = x-y
    d = norm(r)
    RRT = r*transpose(r) # r ⊗ rᵗ
    drdn    = vdot(r,ny)/d
    if N==2
        ID = Mat2{Float64}(1,0,0,1)
        return μ/(2π*(1-ν)*d^2)* (2*drdn/d*( (1-2ν)*nx*transpose(r) + ν*(vdot(r,nx)*ID + r*transpose(nx)) - 4*vdot(r,nx)*RRT/d^2 ) +
                                  2*ν/d^2*(vdot(r,nx)*ny*transpose(r) + vdot(nx,ny)*RRT) +
                                  (1-2*ν)*(2/d^2*vdot(r,nx)*r*transpose(ny) + vdot(nx,ny)*ID + ny*transpose(nx)) -
                                  (1-4ν)*nx*transpose(ny)
                                  )
    elseif N==3
        ID = Mat3{Float64}(1,0,0,0,1,0,0,0,1)
        return μ/(4π*(1-ν)*d^3)* (3*drdn/d*( (1-2ν)*nx*transpose(r) + ν*(vdot(r,nx)*ID + r*transpose(nx)) - 5*vdot(r,nx)*RRT/d^2 ) +
                                  3*ν/d^2*(vdot(r,nx)*ny*transpose(r) + vdot(nx,ny)*RRT) +
                                  (1-2*ν)*(3/d^2*vdot(r,nx)*r*transpose(ny) + vdot(nx,ny)*ID + ny*transpose(nx)) -
                                  (1-4ν)*nx*transpose(ny)
                                  )
    end
end
HypersingularKernel{N}(op::Op,args...) where {N,Op<:Elastostatic} = HypersingularKernel{N,Mat{N,N,Float64,N*N},Op}(op,args...)
