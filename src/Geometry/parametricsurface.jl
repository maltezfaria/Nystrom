"""
    ParametricSurface{M,N,T}

Representation of an `M` dimensional parametric surface embedded in `M` dimensional space.

The parametrization takes `M` parametric coordinates and gives a `Point{N,T}`
representing that parametric coordinate in real space. Typically `M` takes one of the following values:
    - `M=1` represents a line
    - `M=2` represents a surface
    - `M=3` represents a volume

By default a hypersurface is assumed, so that if `N` is the only parameter
given, the default constructor sets `M=N-1` and `T=Float64`.
"""
struct ParametricSurface{M,N,T}
    parametrization::Function
    domain::Cuboid{M,T}
    elements::Vector{Cuboid{M,T}}
end

ParametricSurface(surf,domain::Cuboid{M,T}) where {M,T} = ParametricSurface{M,M+1,Float64}(surf,domain,[domain])

"""
    GmshParametricSurface{M}

An `M` dimensional entity represented by `GMSH` through its dimension `M` and `tag`.

Behaves similar to `ParametricSurface`, but since we do not have direct access
to the function representation of the underlying entity different methods have
pto be called to query the surface for information.

Although this is useful for dealing with more complex CAD surfaces, such objects
depend on the external library `gmsh` and therefore the validity cannot be verified
internally (e.g. other methods could retag the surface, or close `gmsh`).

See also: [`ParametricSurface`](@ref)
"""
struct GmshParametricSurface{M}
    # dim=M
    tag::Int
    elements::Vector{Cuboid{M,Float64}}
end

ambient_dim(p::ParametricSurface{M,N}) where {M,N}   = N
ambient_dim(p::GmshParametricSurface)                = 3 #everything in gmsh is in 3d
geometric_dim(p::ParametricSurface{M}) where {M}     = M
geometric_dim(p::GmshParametricSurface{M}) where {M} = M

ParametricSurface{N}(args...) where {N} = ParametricSurface{N-1,N,Float64}(args...)

function GmshParametricSurface(dim::Int,tag::Int,model=gmsh.model.getCurrent())
    (umin,vmin),(umax,vmax) = gmsh.model.getParametrizationBounds(dim,tag)
    rec = Cuboid(umin,vmin,umax-umin,vmax-vmin)
    return GmshParametricSurface{dim}(tag,model,[rec])
end

getelements(surf::ParametricSurface) = surf.elements

(par::ParametricSurface)(x) = par.parametrization(x)

function (par::GmshParametricSurface{N})(x) where {N}
    if N === 1
        return gmsh.model.getValue(N,par.tag,x)
    elseif N===2
        return gmsh.model.getValue(N,par.tag,[x[1],x[2]])
    else
        error("got N=$N, values must be 1 or 2")
    end
end

jacobian(psurf::ParametricSurface,s::AbstractArray) = ForwardDiff.jacobian(psurf,s)
jacobian(psurf::ParametricSurface,s)                = ForwardDiff.jacobian(psurf,[s...])

function jacobian(psurf::GmshParametricSurface{N},s::Point) where {N}
    if N==1
        jac = gmsh.model.getDerivative(N,psurf.tag,s)
        return reshape(jac,3,N)
    elseif N==2
        jac = gmsh.model.getDerivative(N,psurf.tag,[s[1],s[2]])
        return reshape(jac,3,N)
    else
        error("got N=$N, values must be 1 or 2")
    end
end

"""
    refine!(surf::ParametricSurface)
"""
function refine!(surf::ParametricSurface,ielem,axis)
    elem      = surf.elements[ielem]
    mid_point = (elem.low_corner[axis]+elem.high_corner[axis])/2
    elem1, elem2    = split(elem, axis, mid_point)
    surf.elements[ielem] = elem1
    push!(surf.elements,elem2)
    return  surf
end
function refine!(surf::ParametricSurface{M},ielem) where {M}
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

"""
    meshgen!(surf::ParametricSurface,h)
"""
function meshgen!(surf::ParametricSurface{M},h) where {M}
    domain = surf.domain
    n      = ceil.(Int,(domain.high_corner-domain.low_corner) / h)
    elements = surf.elements
    resize!(elements,prod(n))
    if M == 1
        x = range(domain.low_corner[1],domain.high_corner[1],length=n[1]+1)
        for i in 1:n[1]
            elements[i] = Cuboid(x[i],x[i+1])
        end
    elseif M == 2
        x = range(domain.low_corner[1],domain.high_corner[1],length=n[1]+1)
        y = range(domain.low_corner[2],domain.high_corner[2],length=n[2]+1)
        cc = 1
        for i in 1:n[1]
            for j in 1:n[2]
                elements[cc] = Cuboid((x[i],y[j]),(x[i+1],y[j+1]))
                cc += 1
            end
        end
    else
        error("type parameter $M must be 1 or 2")
    end
    return surf
end

#refine all elements in all directions
function refine!(surf::ParametricSurface)
    n = length(surf.elements)
    for i in 1:n
        refine!(surf,i)
    end
end

################################################################################
## PLOTTING RECIPES
################################################################################
@recipe function f(surf::ParametricSurface{1,2}) # curves
    legend --> false
    grid   --> false
    aspect_ratio --> :equal
    elements = surf.elements
    for n in 1:length(elements)
        el = elements[n]
        @series begin
            linecolor --> n
            pts = surf.(getpoints(el))
            x = [pt[1] for pt in pts]
            y = [pt[2] for pt in pts]
            x,y
        end
    end
end

@recipe function f(surf::ParametricSurface{2,3}) # surfaces
    legend --> false
    grid   --> false
    # aspect_ratio --> :equal
    seriestype := :surface
    # color  --> :blue
    linecolor --> :black
    elements = surf.elements
    for n in 1:length(elements)
        el = elements[n]
        @series begin
            fillcolor --> n
            pts = surf.(getpoints(el))
            x = [pt[1] for pt in pts]
            y = [pt[2] for pt in pts]
            z = [pt[3] for pt in pts]
            x,y,z
        end
    end
end
