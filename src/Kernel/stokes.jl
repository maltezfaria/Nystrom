############################ STOKES ############################3
struct Stokes{N,T} <: AbstractPDE{N}
    μ::T
end
Stokes(;μ,dim=3)             = Stokes{dim}(μ)
Stokes{N}(μ::T) where {N,T}  = Stokes{N,T}(μ)

getname(::Stokes) = "Stokes"

default_kernel_eltype(::Stokes{N}) where {N} = Mat{N,N,Float64,N*N}
default_density_eltype(::Stokes{N}) where {N} = Vec{N,Float64}

# Single Layer
function (SL::SingleLayerKernel{T,S})(x,y)::T  where {T,S<:Stokes}
    N = ambient_dim(S)
    x==y && return zero(T)
    μ = SL.op.μ
    r = x .- y
    d = norm(r)
    RRT = r*transpose(r)
    if N==2
        ID = Mat{2,2,Float64,4}(1,0,0,1)
        γ  = -log(d)
    elseif N==3
        ID = Mat3{Float64}(1,0,0,0,1,0,0,0,1)
        γ  = 1/d
   end
    return 1/(4π*(N-1)*μ) * (γ*ID + RRT/d^N)
end

# Double Layer Kernel
function (DL::DoubleLayerKernel{T,S})(x,y,ny)::T where {T,S<:Stokes}
    N = ambient_dim(S)
    μ = DL.op.μ
    r = x .- y
    d = norm(r)
    if N==2
        x==y && return zero(T)
        return 1/π * dot(r,ny)/d^4 .* r*transpose(r)
    elseif N==3
        x==y && return zero(T)
        ID = Mat3{Float64}(1,0,0,0,1,0,0,0,1)
        RRT = r*transpose(r) # r ⊗ rᵗ
        return 3/(4π) * dot(r,ny)/d^5 .* RRT
    end
end

# TODO: code the hypersingular and adjoint double layer operators for Stokes in 2 and 3d
# Adjoint Double Layer Kernel
# function (HS::AdjointDoubleLayerKernel{N,T,Op})(x,y,ny)::T where {N,T,Op<:Stokes}
#     μ = HS.op.μ
#     r = x .- y
#     d = norm(r)
#     if N==2
#         x==y && return zero(T)
#         return -1/π * dot(r,ny)/d^4 .* r*transpose(r)
#     elseif N==3
#         x==y && return zero(T)
#         ID = Mat3{Float64}(1,0,0,0,1,0,0,0,1)
#         RRT = r*transpose(r) # r ⊗ rᵗ
#         return -3/(4π) * dot(r,ny)/d^5 .* RRT
#     end
# end
# AdjointDoubleLayerKernel{N}(op::Op,args...) where {N,Op<:Stokes} = AdjointDoubleLayerKernel{N,Mat{N,N,Float64,N*N},Op}(op,args...)

# # Hypersingular
# function (DL::HyperSingularKernel{N,T,Op})(x,y,nx,ny)::T where {N,T,Op<:Stokes}
#     μ = DL.op.μ
#     r = x .- y
#     d = norm(r)
#     if N==2
#         x==y && return zero(T)
#         return -1/π * dot(r,ny)/d^4 .* r*transpose(r)
#     elseif N==3
#         x==y && return zero(T)
#         ID = Mat3{Float64}(1,0,0,0,1,0,0,0,1)
#         RRT = r*transpose(r) # r ⊗ rᵗ
#         return -3/(4π) * dot(r,ny)/d^5 .* RRT
#     end
# end
# HyperSingularKernel{N}(op::Op,args...) where {N,Op<:Stokes} = HyperSingularKernel{N,Mat{N,N,Float64,N*N},Op}(op,args...)
