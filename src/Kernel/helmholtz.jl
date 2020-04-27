################################################################################
## HELMHOLTZ
################################################################################

struct Helmholtz{N,T} <: AbstractPDE{N}
    k::T
end
Helmholtz(;k,dim=3) = Helmholtz{dim,typeof(k)}(k)

getname(::Helmholtz) = "Helmholtz"

default_kernel_type(::Helmholtz) = ComplexF64
default_density_type(::Helmholtz{N}) where {N} = ComplexF64

# Single Layer
function (SL::SingleLayerKernel{T,S})(x,y)::T  where {T,S<:Helmholtz}
    N = ambient_dim(S)
    x==y && return zero(T)
    k = SL.op.k
    r = x .- y
    d = sqrt(sum(r.^2))
    # d = norm(x-y)
    if N==2
        return im/4 * hankelh1(0,k*d)
    elseif N==3
        return 1/(4π)/d * exp(im*k*d)
    end
end

# Double Layer Kernel
function (DL::DoubleLayerKernel{T,S})(x,y,ny)::T where {T,S<:Helmholtz}
    N = ambient_dim(S)
    x==y && return 0
    k = DL.op.k
    r = x .- y
    d = sqrt(sum(r.^2))
    # d = norm(x-y)
    if N==2
        return im*k/4/d * hankelh1(1,k*d) .* dot(r,ny)
    elseif N==3
        return 1/(4π)/d^2 * exp(im*k*d) * ( -im*k + 1/d ) * dot(r,ny)
    end
end

# Adjoint double Layer Kernel
function (ADL::AdjointDoubleLayerKernel{T,S})(x,y,nx)::T where {T,S<:Helmholtz}
    N = ambient_dim(S)
    x==y && return 0
    k = ADL.op.k
    r = x .- y
    d = norm(r)
    if N==2
        return -im*k/4/d * hankelh1(1,k*d) .* dot(r,nx)
    elseif N==3
        return -1/(4π)/d^2 * exp(im*k*d) * ( -im*k + 1/d ) * dot(r,nx)
    end
end

# Hypersingular kernel
function (HS::HyperSingularKernel{T,S})(x,y,nx,ny)::T where {T,S<:Helmholtz}
    N = ambient_dim(S)
    x==y && return 0
    k = HS.op.k
    r = x .- y
    d = norm(x-y)
    if N==2
        ID = Mat{2,2,Float64,4}(1,0,0,1)
        RRT = r*transpose(r) # r ⊗ rᵗ
        # TODO: rewrite the operation below in a more clear/efficient way
        return transpose(nx)*((-im*k^2/4/d^2*hankelh1(2,k*d).*RRT + im*k/4/d*hankelh1(1,k*d).*ID)*ny)
    elseif N==3
        ID = Mat{3,3,Float64,9}(1,0,0,0,1,0,0,0,1)
        RRT = r*transpose(r) # r ⊗ rᵗ
        term1 = 1/(4π)/d^2 * exp(im*k*d) * ( -im*k + 1/d ) * ID
        term2 = RRT/d * exp(im*k*d)/(4*π*d^4) * (3*(d*im*k-1) + d^2*k^2)
        return  transpose(nx)*(term1 + term2) * ny
    end
end
