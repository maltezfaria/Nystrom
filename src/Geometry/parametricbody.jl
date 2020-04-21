"""
    ParametricBody{M,N,T}

A very simple (parametric) representation of a geometrical object as given by a union of `patches` of type
`ParametricSurface{M,N,T}`.

See also: [`ParametricSurface`](@ref)
"""
struct ParametricBody{M,N,T}
    patches::Vector{ParametricSurface{M,N,T}}
end

ParametricBody{N}(args...) where {N} = ParametricBody{N-1,N,Float64}(args...)

function refine!(geo::ParametricBody)
    for patch in geo.patches
        refine!(patch)
    end
    return geo
end



