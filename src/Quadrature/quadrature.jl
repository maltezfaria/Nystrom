"""
    Quadrature{Q,N,T}

Basic quadrature structure for integration over an object in `R^N`.

The type parameter `Q` is used to encode information about the type of
quadrature used (e.g. quadrature rule). `Q=Any` means no information is
provided.

The `elements` field contains in its `i` entry the indices of the nodes which
form the `i` element, in case such an element structure is present. This is
sometimes useful for routines which iterating over the `elements` instead of just the `nodes`.
"""
struct Quadrature{Q,N,T}
    nodes::Vector{Point{N,T}}
    normals::Vector{Normal{N,T}}
    weights::Vector{T}
    elements::Vector{Vector{Int}}
end

"""
    TensorQuadrature{S}

A singleton type representing a quadrature of size `S`, where `S` is usually a tuple.
"""
struct TensorQuadrature{S} end
#NOTE: figure out how to restrict the type parameter `S` above to reinforce it is `NTuple{N,Int} where {N}`

Base.length(::Type{TensorQuadrature{S}}) where {S} = prod(S)
Base.size(::Type{TensorQuadrature{S}}) where {S} = S

"""
    _rescale_quadrature(origin,width,algo)

Convenience method to rescale the `p` points 1d quadrature rule `algo` from
`[-1,1]` to an interval given by `[origin,origin+width]`
"""
function _rescale_quadrature(p,origin,width,algo)
    nodes, weights = algo(p)
    @. nodes       = nodes*width/2            # shift to [-w/2,w/2]
    @. nodes       = nodes + width/2 + origin #shift to [a,b]
    @. weights     = weights * width/2
    return nodes, weights
end

function tensorquadrature(el::HyperRectangle{N},p,algo) where {N}
    nodes1d   = Vector{Vector{Float64}}(undef,N)
    weights1d = Vector{Vector{Float64}}(undef,N)
    for n in 1:N
        nodes1d[n], weights1d[n] = _rescale_quadrature(p[n],el.origin[n],el.widths[n],algo)
    end
    nodes = Vector{Point{N,Float64}}(undef,prod(p))
    for (n,node) in enumerate(Iterators.product(nodes1d...))
        nodes[n] = Point(node)
    end
    weights = Array{Float64,N}(undef,p...)
    for (n,weight) in enumerate(Iterators.product(weights1d...))
        weights[n] = prod(weight)
    end
    return nodes,weights
end

function tensorquadrature(geo::ParametricBody{M,N,T},p) where {M,N,T}
    quad = Quadrature{N,T}([],[],[],p^(N-1))
    for surf in patches
        for el in surf.elements
            el_nodes, el_normals, el_weights = quadgen(surf,el,p)
            push!(quad.nodes,el_nodes...)
            push!(quad.weights,el_weights...)
            push!(quad.normals,el_normals...)
        end
    end
    return quad
end

function tensorquadrature(surf::ParametricSurface{M,N,T}, p, algo=gausslegendre) where {M,N,T}
    nelements = length(elements(surf))
    nnodes    = nelements*prod(p)
    # preallocate vectors
    nodes     = Vector{Point{N,T}}(undef,nnodes)
    normals   = Vector{Normal{N,T}}(undef,nnodes)
    weights   = Vector{T}(undef,nnodes)
    # compute quadrature on reference element. Note: this assumes the reference elements are the same
    ref_nodes, ref_weights = tensorquadrature(element,p,algo) # quadrature in reference element
    quad_elements = Vector{Int}[]
    inode = 1
    iel   = 1
    for element in  elements(surf)
        for (node,weight) in zip(ref_nodes,ref_weights)
            nodes[inode] = surf(node)
            push!(quad_elements[iel],inode)
            jac      = jacobian(surf,node)
            if N==2
                jac_det    = norm(jac)
                normals[inode] = [jac[2],-jac[1]]./jac_det
            elseif N==3
                tmp = cross(jac[:,1],jac[:,2])
                jac_det = norm(tmp)
                normals[inode] = tmp./jac_det
            end
            weights[inode] = jac_det*weight
            inode +=1
        end
        iel += 1
    end
    # create a quadrature type tag
    Q = TensorQuadrature{p}
    return Quadrature{Q,3,T}(nodes,weights,normals,quad_elements)
end

# quadrature for (flat) hyper-rectangle
function quadgen(domain::HyperRectangle{N},p) where {N}
    N == 1 && return quad_interval(p,domain)
    N == 2 && return quad_square(p,domain)
end

function quad_1d(p,origin,width)
    nodes, weights = gausslegendre(p)
    #nodes, weights = newtoncotes(p)
    @. nodes       = nodes*width/2  # shift to [-w/2,w/2]
    @. nodes       = nodes + width/2 + origin #shift to [a,b]
    @. weights = weights * width/2
    return nodes, weights
end

function quad_interval(p,hr::HyperRectangle{1,T}) where {T}
    nodes, weights = quad_1d(p,hr.origin,hr.widths)
    return Vector{T}(nodes), Vector{T}(weights)
end

function quad_square(p,hr::HyperRectangle{2,T}) where {T}
    nodes_x, weights_x  = quad_1d(p,hr.origin[1],hr.widths[1])
    nodes_y, weights_y  = quad_1d(p,hr.origin[2],hr.widths[2])
    nodes    = vec([(nx,ny) for nx in nodes_x, ny in nodes_y])
    weights  = vec([ wx*wy for wx in weights_x, wy in weights_y])
    return nodes, weights
end
