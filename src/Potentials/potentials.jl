struct SurfaceDensity{V,S} <: AbstractVector{V}
    vals::Vector{V}
    surface::S
end
Base.size(σ::SurfaceDensity,args...)     = size(σ.vals,args...)
Base.getindex(σ::SurfaceDensity,args...) = getindex(σ.vals,args...)
Base.setindex(σ::SurfaceDensity,args...) = setindex(σ.vals,args...)

SurfaceDensity(etype::DataType,surf) = SurfaceDensity(zeros(etype,length(surf)),surf)

# function gmres!(σ::SurfaceDensity,A,μ::SurfaceDensity,args...;kwargs...)
#     gmres!(σ.vals,A,μ.vals,args...;kwargs...)
#     return σ
# end

function γ₀(f,X)
    vals = [f(x) for x in getnodes(X)]
    return SurfaceDensity(vals,X)
end

#FIXME
function ∇ end

function γ₁(f,X)
    vals = [∇(f)(x)⋅n for (x,n) in zip(getnodes(X),getnormals(X))]
    return SurfaceDensity(vals,X)
end

struct IntegralPotential{T<:AbstractKernel,S}
    kernel::T
    surface::S
end

function IntegralPotential(op::AbstractPDE,surf,ktype::Symbol)
    if ktype == :singlelayer
        k = SingleLayerKernel(op)
    elseif ktype == :doublelayer
        k = DoubleLayerKernel(op)
    else
        error("unknown kernel type: valid options are :singlelayer or :doublelayer")
    end
    return IntegralPotential(k,surf)
end

kernel_type(pot::IntegralPotential) = kernel_type(pot.kernel)

(pot::IntegralPotential)(σ::AbstractVector,x) = pot(kernel_type(pot),σ,x)
function (pot::IntegralPotential)(::SingleLayer,σ::AbstractVector,x)
    f = pot.kernel
    Γ = pot.surface
    iter = zip(getnodes(Γ),getweights(Γ),σ)
    out = mapreduce(+,iter) do (y,w,σ)
        f(x,y)*σ*w
    end
    return out
end
function (pot::IntegralPotential)(::DoubleLayer,σ::AbstractVector,x)
    f = pot.kernel
    Y = pot.surface
    iter = zip(getnodes(Y),getweights(Y),getnormals(Y),σ.vals)
    out  = mapreduce(+,iter) do (y,w,ny,σ)
        f(x,y,ny)*σ*w
    end
    return out
end

Base.getindex(pot::IntegralPotential,σ::SurfaceDensity) = (x) -> pot(σ,x)

SingleLayerPotential(op::AbstractPDE,surf) = IntegralPotential(SingleLayerKernel(op),surf)
DoubleLayerPotential(op::AbstractPDE,surf) = IntegralPotential(DoubleLayerKernel(op),surf)

# function (pot::SingleLayerPotential)(σ::AbstractVector,x)
#     f = pot.kernel
#     Γ = pot.surface
#     iter = zip(getnodes(Γ),getweights(Γ),σ)
#     out = mapreduce(+,iter) do (y,w,σ)
#         f(x,y)*σ*w
#     end
#     return out
# end

# struct DoubleLayerPotential{T,S}
#     kernel::T
#     surface::S
# end

# function (pot::DoubleLayerPotential)(σ::AbstractVector,x)
#     f = pot.kernel
#     Y = pot.Y
#     iter = zip(getnodes(Y),getweights(Y),getnormals(Y),σ.vals)
#     out  = mapreduce(+,iter) do (y,w,ny,σ)
#         f(x,y,ny)*σ*w
#     end
#     return out
# end



