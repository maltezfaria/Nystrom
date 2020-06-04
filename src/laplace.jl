################################################################################
## LAPLACE
################################################################################
struct Laplace{N} <: AbstractPDE{N} end
Laplace(;dim=3) = Laplace{dim}()

getname(::Laplace) = "Laplace"

default_kernel_eltype(::Laplace) = Float64
default_density_eltype(::Laplace{N}) where {N} = Float64

# Single Layer
function (SL::SingleLayerKernel{T,Laplace{N}})(x,y)::T  where {N,T}
    x==y && return zero(T)
    r = x .- y
    d = sqrt(sum(r.^2))
    if N==2
        return -1/(2π)*log(d)
    elseif N==3
        return 1/(4π)/d
    end
end

# Double Layer Kernel
function (DL::DoubleLayerKernel{T,Laplace{N}})(x,y,ny)::T where {N,T}
    x==y && return zero(T)
    r = x .- y
    d = sqrt(sum(r.^2))
    if N==2
        return 1/(2π)/(d^2) .* dot(r,ny)
    elseif N==3
        return 1/(4π)/(d^3) .* dot(r,ny)
    end
end

# Adjoint double Layer Kernel
function (ADL::AdjointDoubleLayerKernel{T,Laplace{N}})(x,y,nx)::T where {N,T}
    x==y && return zero(T)
    r = x .- y
    d = sqrt(sum(r.^2))
    if N==2
        return -1/(2π)/(d^2) .* dot(r,nx)
    elseif N==3
        return -1/(4π)/(d^3) .* dot(r,nx)
    end
end

# Hypersingular kernel
function (HS::HyperSingularKernel{T,Laplace{N}})(x,y,nx,ny)::T where {N,T}
    x==y && return zero(T)
    r = x .- y
    d = sqrt(sum(r.^2))
    if N==2
        ID = Mat{2,2,Float64,4}(1,0,0,1)
        RRT = r*transpose(r) # r ⊗ rᵗ
        return 1/(2π)/(d^2) * transpose(nx)*(( ID -2*RRT/d^2  )*ny)
    elseif N==3
        ID = Mat{3,3,Float64,9}(1,0,0,0,1,0,0,0,1)
        RRT = r*transpose(r) # r ⊗ rᵗ
        return 1/(4π)/(d^3) * transpose(nx)*(( ID -3*RRT/d^2  )*ny)
    end
end

