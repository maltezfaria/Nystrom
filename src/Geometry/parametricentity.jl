"""
    AbstractEntity{M,N,T}

Representation of an `M` dimensional parametric entity embedded in `M` dimensional space.

The parametrization takes `M` parametric coordinates and gives a `Point{N,T}`, where `M ≤ N`, 
representing that parametric coordinate in real space. `M` is the geometrical dimension of the object:
    - `M=1` --> line
    - `M=2` --> surface
    - `M=3` --> volume
"""
abstract type AbstractEntity{M,N,T} end

ambient_dim(p::AbstractEntity{M,N}) where {M,N}   = N
geometric_dim(p::AbstractEntity{M}) where {M}     = M

getelements(surf::AbstractEntity) = surf.elements

struct ParametricCurve{N,T} <: AbstractEntity{1,N,T}
    parametrization
    elements::Vector{HyperRectangle{1,T}}
    boundary::Vector{Point{1,T}}
end

struct ParametricSurface{N,T} <: AbstractEntity{2,N,T}
    parametrization
    elements::Vector{HyperRectangle{2,T}}
    boundary::Vector{ParametricCurve{N,T}}
end

struct ParametricVolume{N,T} <: AbstractEntity{3,N,T}
    parametrization
    elements::Vector{HyperRectangle{3,T}}
    boundary::Vector{ParametricSurface{N,T}}
end

# """
#     GmshParametricSurface{M}

# An `M` dimensional entity represented by `GMSH` through its dimension `M` and `tag`.

# Behaves similar to `ParametricSurface`, but since we do not have direct access
# to the function representation of the underlying entity different methods have
# to be called to query the surface for information.

# Although this is useful for dealing with more complex CAD surfaces, such objects
# depend on the external library `gmsh` and therefore the validity cannot be verified
# internally (e.g. other methods could retag the surface, or close `gmsh`).

# See also: [`ParametricSurface`](@ref)
# """

struct GmshParametricCurve <: AbstractEntity{1,3,Float64}
    tag::Int
    elements::Vector{HyperRectangle{1,Float64}}
    boundary::Vector{Point{3,Float64}}
end

struct GmshParametricSurface <: AbstractEntity{2,3,Float64}
    tag::Int
    elements::Vector{HyperRectangle{2,Float64}}
    boundary::Vector{GmshParametricCurve}
end

struct GmshParametricVolume <: AbstractEntity{2,3,Float64}
    # dim=M
    tag::Int
    elements::Vector{HyperRectangle{3,Float64}}
    boundary::Vector{GmshParametricSurface}
end

(par::AbstractEntity)(x)        = par.parametrization(x)

(par::GmshParametricCurve)(x)   = gmsh.model.getValue(1,par.tag,x)
(par::GmshParametricSurface)(x) = gmsh.model.getValue(2,par.tag,x)

jacobian(psurf::AbstractEntity,s::AbstractArray) = ForwardDiff.jacobian(psurf,s)
jacobian(psurf::AbstractEntity,s)                = ForwardDiff.jacobian(psurf,[s...])

function jacobian(psurf::GmshParametricCurve,s::Point)
    jac = gmsh.model.getDerivative(N,psurf.tag,s)
    return reshape(jac,3,1)
end

function jacobian(psurf::GmshParametricSurface,s::Point)
    jac = gmsh.model.getDerivative(N,psurf.tag,[s[1],s[2]])
    return reshape(jac,3,2)
end

"""
    refine!(surf::ParametricSurface)
"""
function refine!(surf::AbstractEntity,ielem,axis)
    elem      = surf.elements[ielem]
    mid_point = elem.origin[axis]+elem.widths[axis]/2
    elem1, elem2    = split(elem, axis, mid_point)
    surf.elements[ielem] = elem1
    push!(surf.elements,elem2)
    return  surf
end
function refine!(surf::AbstractEntity{M},ielem) where {M}
    if M == 1
        refine!(surf,ielem,1)
    elseif M == 2
        refine!(surf,ielem,1)
        n = length(getelements(surf))
        refine!(surf,ielem,2)
        refine!(surf,n,2)
    else
        @error "method not implemented"
    end
    return surf
end

#refine all elements in all directions
function refine!(surf::AbstractEntity)
    n = length(surf.elements)
    for i in 1:n
        refine!(surf,i)
    end
end

function tensorquadrature(q::TensorQuadrature,surf::AbstractEntity{M,N,T},algo) where {M,N,T}
    Q = typeof(q)
    quad = Quadrature{Q,N,T}()
    for element in  getelements(surf)
        # compute quadrature on reference element
        push!(quad.elements,[])
        ref_quad = tensorquadrature(q,element,algo) # quadrature in reference element
        for (node,weight) in zip(getnodes(ref_quad),getweights(ref_quad))
            push!(quad.nodes,surf(node))
            push!(quad.elements[end],length(quad.nodes))
            jac      = jacobian(surf,node)
            if N==2
                jac_det    = norm(jac)
                normal = [jac[2],-jac[1]]./jac_det
            elseif N==3
                tmp = cross(jac[:,1],jac[:,2])
                jac_det = norm(tmp)
                normal = tmp./jac_det
            end
            push!(quad.normals,normal)
            push!(quad.weights,jac_det*weight)
        end
    end
    return quad
end

################################################################################
## SIMPLE SHAPES
################################################################################

# 2D
function ellipsis(;paxis=ones(2),center=ones(2))
    f(s)       = center .+ paxis.*[cospi(s[1]),sinpi(s[1])]
    domain     = HyperRectangle(-1.0,2.0)
    surf       = ParametricSurface{2}(f,[domain])
    return ParametricBody{2}([surf])
end
circle(;radius=1,center=ones(2)) = ellipsis(;paxis=radius*ones(2),center=center)

function kite(;radius=1,center=ones(2))
    f(s) = center .+ rad.*[cospi(s[1]) + 0.65*cospi(2*s[1]) - 0.65,
                1.5*sinpi(s[1])]
    domain = HyperRectangle(-1.0,2.0)
    surf   = ParametricSurface{2}(f,[domain])
    return ParametricBody{2}([surf])
end

# 3D
function cube(;paxis=ones(3),center=zeros(3))
    nparts = 6
    domain = HyperRectangle(-1.,-1.,2.,2.)
    parts = ParametricSurface{2,3,Float64}[]
    for id=1:nparts
        param(x) = _cube_parametrization(x[1],x[2],id,paxis,center)
        patch = ParametricSurface{3}(param,[domain])
        push!(parts,patch)
    end
    return ParametricBody{3}(parts)
end

function ellipsoid(;paxis=ones(3),center=zeros(3))
    nparts = 6
    domain = HyperRectangle(-1.,-1.,2.,2.)
    parts = ParametricSurface{2,3,Float64}[]
    for id=1:nparts
        param(x) = _ellipsoid_parametrization(x[1],x[2],id,paxis,center)
        patch = ParametricSurface{3}(param,[domain])
        push!(parts,patch)
    end
    return ParametricBody{3}(parts)
end
sphere(;radius=1,center=zeros(3)) = ellipsoid(;paxis=radius*ones(3),center=center)

function bean(;paxis=ones(3),center=zeros(3))
    nparts = 6
    domain = HyperRectangle(-1.,-1.,2.,2.)
    parts  = ParametricSurface{2,3,Float64}[]
    for id=1:nparts
        param(x) = _bean_parametrization(x[1],x[2],id,paxis,center)
        patch    = ParametricSurface{3}(param,[domain])
        push!(parts,patch)
    end
    return ParametricBody{3}(parts)
end

function _cube_parametrization(u,v,id,paxis,center)
    if id==1
        x = [1.,u,v]
    elseif id==2
        x = [-u,1.,v];
    elseif id==3
        x = [u,v,1.];
    elseif id==4
        x =[-1.,-u,v];
    elseif id==5
        x = [u,-1.,v];
    elseif id==6
        x = [-u,v,-1.];
    end
    return center .+ paxis.*x
end


function _sphere_parametrization(u,v,id,rad=1,center=zeros(3))
    if id==1
        x = [1.,u,v]
    elseif id==2
        x = [-u,1.,v];
    elseif id==3
        x = [u,v,1.];
    elseif id==4
        x =[-1.,-u,v];
    elseif id==5
        x = [u,-1.,v];
    elseif id==6
        x = [-u,v,-1.];
    end
    return center .+ rad.*x./sqrt(u^2+v^2+1)
end

function _ellipsoid_parametrization(u,v,id,paxis=ones(3),center=zeros(3))
    x = _sphere_parametrization(u,v,id)
    return x .* paxis .+ center
end

function _bean_parametrization(u,v,id,paxis=one(3),center=zeros(3))
    x = _sphere_parametrization(u,v,id)
    a = 0.8; b = 0.8; alpha1 = 0.3; alpha2 = 0.4; alpha3=0.1
    x[1] = a*sqrt(1.0-alpha3*cospi(x[3])).*x[1]
    x[2] =-alpha1*cospi(x[3])+b*sqrt(1.0-alpha2*cospi(x[3])).*x[2];
    x[3] = x[3];
    return x .* paxis .+ center
end

n
