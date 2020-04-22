abstract type AbstractPDE{N} end

Base.ndims(op::AbstractPDE{N}) where {N}       = N
Base.ndims(::Type{<:AbstractPDE{N}}) where {N} = N

abstract type AbstractKernel{T} end
return_type(K::AbstractKernel{T}) where {T} = T

struct SingleLayerKernel{T,Op} <: AbstractKernel{T}
    op::Op
end
SingleLayerKernel{T}(op) where {T} = SingleLayerKernel{T,typeof(op)}(op)
SingleLayerKernel(op)              = SingleLayerKernel{default_kernel_type(op)}(op)

struct DoubleLayerKernel{T,Op} <: AbstractKernel{T}
    op::Op
end
DoubleLayerKernel{T}(op) where {T} = DoubleLayerKernel{T,typeof(op)}(op)
DoubleLayerKernel(op)              = DoubleLayerKernel{default_kernel_type(op)}(op)

struct AdjointDoubleLayerKernel{T,Op} <: AbstractKernel{T}
    op::Op
end
AdjointDoubleLayerKernel{T}(op) where {T} = AdjointDoubleLayerKernel{T,typeof(op)}(op)
AdjointDoubleLayerKernel(op)              = AdjointDoubleLayerKernel{default_kernel_type(op)}(op)

struct HyperSingularKernel{T,Op} <: AbstractKernel{T}
    op::Op
end
HyperSingularKernel{T}(op) where {T} = HyperSingularKernel{T,typeof(op)}(op)
HyperSingularKernel(op)              = HyperSingularKernel{default_kernel_type(op)}(op)

# kernel trait do distinguish various integral operators and potentials. This helps in dispatch.
struct SingleLayer end
struct DoubleLayer end
struct AdjointDoubleLayer end
struct HyperSingular end

kernel_type(::SingleLayerKernel)        = SingleLayer()
kernel_type(::DoubleLayerKernel)        = DoubleLayer()
kernel_type(::AdjointDoubleLayerKernel) = AdjointDoubleLayer()
kernel_type(::HyperSingularKernel)      = HyperSingular()
