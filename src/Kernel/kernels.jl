abstract type AbstractOperator{N} end
Base.ndims(op::AbstractOperator{N}) where {N} = N

abstract type Kernel{T} end
Base.eltype(K::Kernel{T}) where {T} = T

struct SingleLayerKernel{N,T,Op} <: Kernel{N,T}
    op::Op
end
SingleLayerKernel{N,T}(op) where {N,T} = SingleLayerKernel{N,T,typeof(op)}(op)

struct DoubleLayerKernel{N,T,Op} <: Kernel{N,T}
    op::Op
end
DoubleLayerKernel{N,T}(op) where {N,T} = DoubleLayerKernel{N,T,typeof(op)}(op)

struct AdjointDoubleLayerKernel{N,T,Op} <: Kernel{N,T}
    op::Op
end
AdjointDoubleLayerKernel{N,T}(op) where {N,T} = AdjointDoubleLayerKernel{N,T,typeof(op)}(op)

struct HypersingularKernel{N,T,Op} <: Kernel{N,T}
    op::Op
end
HypersingularKernel{N,T}(op) where {N,T} = HypersingularKernel{N,T,typeof(op)}(op)

