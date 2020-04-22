struct SingleLayerPotential{N,Tkernel,Tmesh,Tdensity,Op}
    kernel::SingleLayerKernel{N,Tkernel,Op}
    quad::Quadrature{N,Tmesh}
    density::Vector{Tdensity}
end

function (SL::SingleLayerPotential{N,Tkernel,Tmesh,Tdensity})(x) where {N,Tkernel,Tmesh,Tdensity}
    Tout = Base.promote_op(*,Tkernel,Tdensity)
    out = zero(Tout)
    K   = SL.kernel
    quad = SL.quad
    nodes = quad.nodes
    w     = quad.weights
    φ     = SL.density
    npts = length(SL.quad.weights)
    for i=1:npts
        y = nodes[i]
        out += K(x,y)*φ[i]*w[i]
    end
    return  out
end

struct DoubleLayerPotential{N,Tkernel,Tmesh,Tdensity,Op}
    kernel::DoubleLayerKernel{N,Tkernel,Op}
    quad::Quadrature{N,Tmesh}
    density::Vector{Tdensity}
end

@inline function (DL::DoubleLayerPotential{N,Tkernel,Tmesh,Tdensity})(x) where {N,Tkernel,Tmesh,Tdensity}
    Tout = Base.promote_op(*,Tkernel,Tdensity)
    out = zero(Tout)
    K   = DL.kernel
    quad = DL.quad
    nodes = quad.nodes
    w     = quad.weights
    φ     = DL.density
    npts = length(DL.quad.weights)
    for i=1:npts
        y = nodes[i]
        ny = quad.normals[i]
        out += K(x,y,ny)*φ[i]*w[i]
    end
    return  out
end

