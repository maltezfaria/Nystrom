################################################################################
## ELASTOSTATIC
################################################################################

struct Elastostatic{N,T} <: AbstractPDE{N}
    μ::T
    λ::T
end
Elastostatic(;μ,λ,ndims=3)             = Elastostatic{ndims}(promote(μ,λ)...)
Elastostatic{N}(μ::T,λ::T) where {N,T} = Elastostatic{N,T}(μ,λ)

default_kernel_type(::Elastostatic{N}) where {N} = Mat{N,N,Float64,N*N}
default_density_type(::Elastostatic{N}) where {N} = Vec{N,Float64}

# Single Layer
function (SL::SingleLayerKernel{T,S})(x,y)::T  where {T,S<:Elastostatic}
    N = ndims(S)
    x==y && return zero(T)
    μ = SL.op.μ
    λ = SL.op.λ
    ν = λ/(2*(μ+λ))
    r = x-y
    d = norm(r)
    RRT = r*transpose(r) # r ⊗ rᵗ
    if N==2
        ID = Mat{2,2,Float64,4}(1,0,0,1)
        return 1/(8π*μ*(1-ν))*(-(3-4*ν)*log(d)*ID + RRT/d^2)
    elseif N==3
        ID = Mat{3,3,Float64,9}(1,0,0,0,1,0,0,0,1)
        return 1/(16π*μ*(1-ν)*d)*((3-4*ν)*ID + RRT/d^2)
    end
end

# Double Layer Kernel
function (DL::DoubleLayerKernel{T,S})(x,y,ny)::T where {T,S<:Elastostatic}
    N = ndims(S)
    x==y && return zero(T)
    μ = DL.op.μ
    λ = DL.op.λ
    ν = λ/(2*(μ+λ))
    r = x-y
    d = norm(r)
    RRT = r*transpose(r) # r ⊗ rᵗ
    drdn = dot(r,ny)/d
    if N==2
        ID = Mat2{Float64}(1,0,0,1)
        return -1/(4π*(1-ν)*d)*(drdn*((1-2ν)*ID + 2*RRT/d^2) - (1-2ν)/d*(r*transpose(ny) - ny*transpose(r)))
    elseif N==3
        ID = Mat3{Float64}(1,0,0,0,1,0,0,0,1)
        return -1/(8π*(1-ν)*d^2)*(drdn * ((1-2*ν)*ID + 3*RRT/d^2) - (1-2*ν)/d*(r*transpose(ny) - ny*transpose(r)))
    end
end

# Adjoint Double Layer Kernel
function (ADL::AdjointDoubleLayerKernel{T,S})(x,y,nx)::T where {T,S<:Elastostatic}
    N = ndims(S)
    x==y && return zero(T)
    μ = ADL.op.μ
    λ = ADL.op.λ
    ν = λ/(2*(μ+λ))
    r = x-y
    d = norm(r)
    RRT = r*transpose(r) # r ⊗ rᵗ
    if N==2
        ID = Mat2{Float64}(1,0,0,1)
        return 1/(4π*(1-ν)*d)*((1-2ν)/d*(-nx*transpose(r) + dot(r,nx)*ID + r*transpose(nx)) + 2/d^3*dot(r,nx)*RRT)
    elseif N==3
        ID = Mat3{Float64}(1,0,0,0,1,0,0,0,1)
        return 1/(8π*(1-ν)*d^2)*((1-2ν)/d*(-nx*transpose(r) + dot(r,nx)*ID + r*transpose(nx)) + 3/d^3*dot(r,nx)*RRT)
    end
end

# Hypersingular kernel
function (HS::HyperSingularKernel{T,S})(x,y,nx,ny)::T where {T,S<:Elastostatic}
    x==y && return zero(T)
    μ = HS.op.μ
    λ = HS.op.λ
    ν = λ/(2*(μ+λ))
    r = x-y
    d = norm(r)
    RRT = r*transpose(r) # r ⊗ rᵗ
    drdn    = dot(r,ny)/d
    if N==2
        ID = Mat2{Float64}(1,0,0,1)
        return μ/(2π*(1-ν)*d^2)* (2*drdn/d*( (1-2ν)*nx*transpose(r) + ν*(dot(r,nx)*ID + r*transpose(nx)) - 4*dot(r,nx)*RRT/d^2 ) +
                                  2*ν/d^2*(dot(r,nx)*ny*transpose(r) + dot(nx,ny)*RRT) +
                                  (1-2*ν)*(2/d^2*dot(r,nx)*r*transpose(ny) + dot(nx,ny)*ID + ny*transpose(nx)) -
                                  (1-4ν)*nx*transpose(ny)
                                  )
    elseif N==3
        ID = Mat3{Float64}(1,0,0,0,1,0,0,0,1)
        return μ/(4π*(1-ν)*d^3)* (3*drdn/d*( (1-2ν)*nx*transpose(r) + ν*(dot(r,nx)*ID + r*transpose(nx)) - 5*dot(r,nx)*RRT/d^2 ) +
                                  3*ν/d^2*(dot(r,nx)*ny*transpose(r) + dot(nx,ny)*RRT) +
                                  (1-2*ν)*(3/d^2*dot(r,nx)*r*transpose(ny) + dot(nx,ny)*ID + ny*transpose(nx)) -
                                  (1-4ν)*nx*transpose(ny)
                                  )
    end
end
