abstract type IntegralOperator{N,T} <: AbstractMatrix{T} end

Base.size(IOp::IntegralOperator)      = (length(IOp.quad.nodes), length(IOp.quad.nodes))

function Base.Matrix(IOp::IntegralOperator{N,T}) where {N,T}
    s1,s2 = size(IOp)
    A = Matrix{T}(undef,s1,s2)
    # qsize = IOp.quad.nodes_per_element
    # Threads.@threads for j=1:s2
    #     for i=1:s1
    #         if div(i-1,qsize) == div(j-1,qsize)
    #             A[i,j] = 0
    #         else
    #             A[i,j] = IOp[i,j]
    #         end
    #     end
    # end
    Threads.@threads for j=1:s2
        for i=1:s1
            A[i,j] = IOp[i,j]
        end
    end
    return A
end

# TODO: Find a better way to handle in a generic way vector and scalar-valued problems
# function Base.:*(IOp::IntegralOperator{N,T}, x::AbstractVector{T2}) where {N,T,T2<:Number}
#     T <: Number ? s2 = 1 : s2  = size(T,2)
#     x   = reinterpret(Vec{s2,T2},x)
#     reinterpret(T2,IOp*x)
# end

# # function Base.*(IOp::Integral)

struct SingleLayerOperator{N,T,Tmesh,Op} <: IntegralOperator{N,T}
    kernel::SingleLayerKernel{N,T,Op}
    quad::Quadrature{N,Tmesh}
end
Base.getindex(IOp::SingleLayerOperator,i::Integer,j::Integer)  = IOp.kernel(IOp.quad.nodes[i], IOp.quad.nodes[j])*IOp.quad.weights[j]

struct DoubleLayerOperator{N,T,Tmesh,Op} <: IntegralOperator{N,T}
    kernel::DoubleLayerKernel{N,T,Op}
    quad::Quadrature{N,Tmesh}
end
Base.getindex(IOp::DoubleLayerOperator,i::Integer,j::Integer)  = IOp.kernel(IOp.quad.nodes[i], IOp.quad.nodes[j], IOp.quad.normals[j])*IOp.quad.weights[j]

struct AdjointDoubleLayerOperator{N,T,Tmesh,Op} <: IntegralOperator{N,T}
    kernel::AdjointDoubleLayerKernel{N,T,Op}
    quad::Quadrature{N,Tmesh}
end
Base.getindex(IOp::AdjointDoubleLayerOperator,i::Integer,j::Integer)  = IOp.kernel(IOp.quad.nodes[i], IOp.quad.nodes[j], IOp.quad.normals[i])*IOp.quad.weights[j]

struct HypersingularOperator{N,T,Tmesh,Op} <: IntegralOperator{N,T}
    kernel::HypersingularKernel{N,T,Op}
    quad::Quadrature{N,Tmesh}
end
Base.getindex(IOp::HypersingularOperator,i::Integer,j::Integer)  = IOp.kernel(IOp.quad.nodes[i], IOp.quad.nodes[j], IOp.quad.normals[i],IOp.quad.normals[j])*IOp.quad.weights[j]




