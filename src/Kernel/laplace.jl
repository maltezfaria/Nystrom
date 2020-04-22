############################ LAPLACE ############################
struct Laplace{N} <: AbstractOperator{N} end
Laplace(;dim=3) = Laplace{dim}()
# Single Layer
function (SL::SingleLayerKernel{<:Laplace,T})(x,y)::T  where {T}
    N = ndims(op)
    x==y && return zero(T)
    r = x-y
    d = sqrt(sum(r.^2))
    if N==2
        return -1/(2π)*log(d)
    elseif N==3
        return 1/(4π)/d
    end
end
SingleLayerKernel(op::Laplace,args...) = SingleLayerKernel{Float64,typeof(op)}(op,args...)

# Double Layer Kernel
function (DL::DoubleLayerKernel{N,T,Laplace})(x,y,ny)::T where {N,T}
    x==y && return 0
    r = x-y
    d = sqrt(sum(r.^2))
    if N==2
        return 1/(2π)/(d^2) .* vdot(r,ny)
    elseif N==3
        return 1/(4π)/(d^3) .* vdot(r,ny)
    end
end
DoubleLayerKernel{N}(op::Laplace,args...) where {N} = DoubleLayerKernel{N,Float64,Laplace}(op,args...)

# Adjoint double Layer Kernel
function (ADL::AdjointDoubleLayerKernel{N,T,Laplace})(x,y,nx)::T where {N,T}
    x==y && return 0
    r = x-y
    d = sqrt(sum(r.^2))
    if N==2
        return -1/(2π)/(d^2) .* vdot(r,nx)
    elseif N==3
        return -1/(4π)/(d^3) .* vdot(r,nx)
    end
end
AdjointDoubleLayerKernel{N}(op::Laplace,args...) where {N} = AdjointDoubleLayerKernel{N,Float64,Laplace}(op,args...)

# Hypersingular kernel
function (HS::HypersingularKernel{N,T,Laplace})(x,y,nx,ny)::T where {N,T}
    x==y && return 0
    r = x-y
    d = sqrt(sum(r.^2))
    if N==2
        ID = Mat2{Float64}(1,0,0,1)
        RRT = r*transpose(r) # r ⊗ rᵗ
        return 1/(2π)/(d^2) * transpose(nx)*(( ID -2*RRT/d^2  )*ny)
    elseif N==3
        ID = Mat3{Float64}(1,0,0,0,1,0,0,0,1)
        RRT = r*transpose(r) # r ⊗ rᵗ
        return 1/(4π)/(d^3) * transpose(nx)*(( ID -3*RRT/d^2  )*ny)
    end
end
HypersingularKernel{N}(op::Laplace,args...) where {N} = HypersingularKernel{N,Float64,Laplace}(op,args...)
