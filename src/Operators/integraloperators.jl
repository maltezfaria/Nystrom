"""
    IntegralOperator{T,K,S,V} <: AbstractMatrix{T}

Representation of an integral operator which takes a density ϕ defined on `Y`
and integrates it with `kernel` for all elements `x ∈ X`.
"""
struct IntegralOperator{T,K,S,V} <: AbstractMatrix{T}
    kernel::K
    X::S
    Y::V
end
IntegralOperator{T}(k,X,Y) where {T} = IntegralOperator{T,typeof(k),typeof(X),typeof(Y)}(k,X,Y)
function IntegralOperator(k,X,Y=X)
    T = return_type(k)
    IntegralOperator{T}(k,X,Y)
end

Base.size(iop::IntegralOperator)      = (length(getnodes(iop.X)), length(getnodes(iop.Y)))

kernel_type(iop::IntegralOperator) = kernel_type(iop.kernel)

Base.getindex(iop::IntegralOperator,i::Integer,j::Integer)  = getindex(kernel_type(iop),iop,i,j)

function Base.getindex(::SingleLayer,iop::IntegralOperator,i::Integer,j::Integer)
    x,y,w = getnodes(iop.X)[i], getnodes(iop.Y)[j], getweights(iop.Y)[j]
    return iop.kernel(x,y)*w
end
function Base.getindex(::DoubleLayer,iop::IntegralOperator,i::Integer,j::Integer)
    x,y,ny,w = getnodes(iop.X)[i], getnodes(iop.Y)[j], getnormals(iop.Y)[j], getweights(iop.Y)[j]
    return iop.kernel(x,y,ny)*w
end
function Base.getindex(::AdjointDoubleLayer,iop::IntegralOperator,i::Integer,j::Integer)
    x,y,nx,w = getnodes(iop.X)[i], getnodes(iop.Y)[j], getnormals(iop.X)[i], getweights(iop.Y)[j]
    return iop.kernel(x,y,nx)*w
end
function Base.getindex(::HyperSingular,iop::IntegralOperator,i::Integer,j::Integer)
    x,y,nx,ny,w = getnodes(iop.X)[i], getnodes(iop.Y)[j], getnormals(iop.X)[i], getnormals(iop.Y)[j], getweights(iop.Y)[j]
    return iop.kernel(x,y,nx,ny)*w
end

# convenience constructors
SingleLayerOperator(op::AbstractPDE,X,Y=X)        = IntegralOperator(SingleLayerKernel(op),X,Y)
DoubleLayerOperator(op::AbstractPDE,X,Y=X)        = IntegralOperator(DoubleLayerKernel(op),X,Y)
AdjointDoubleLayerOperator(op::AbstractPDE,X,Y=X) = IntegralOperator(AdjointDoubleLayerKernel(op),X,Y)
HyperSingularOperator(op::AbstractPDE,X,Y=X)      = IntegralOperator(HyperSingularKernel(op),X,Y)
