"""
    Geometry{M,N,T}

A very simple (parametric) representation of a geometrical object as given by a union of `patches` of type
`ParametricSurface{M,N,T}`.

See also: [`ParametricSurface`](@ref)
"""
struct Geometry{M,N,T}
    patches::Vector{ParametricSurface{M,N,T}}
end

Geometry{N}(args...) where {N} = Geometry{M-1,N,Float64}(args...)

function refine!(geo::Geometry)
    for patch in geo.patches
        refine!(patch)
    end
    return geo
end
