"""
    Quadrature{Q,N,T}

Basic quadrature structure for integration over an object in `R^N`.

The type parameter `Q` is used to encode information about the type of
quadrature used (e.g. quadrature rule). 

The `elements` field contains in its `i` entry the indices of the nodes which
form the `i` element, in case such an element structure is present. This is
sometimes useful for routines which iterating over the `elements` instead of just the `nodes`.
"""
struct Quadrature{N,T}
    nodes::Vector{Point{N,T}}
    weights::Vector{T}
    normals::Vector{Normal{N,T}}
    elements::Vector{Vector{Int}}
    bodies::Vector{Vector{Int}}
end
Quadrature{N,T}() where {Q,N,T}= Quadrature{N,T}([],[],[],[],[])

ambient_dim(q::Quadrature{N}) where {Q,N}   = N
geometric_dim(q::Quadrature{N}) where {Q,N} = geometric_dim(Q)

getweights(q::Quadrature)  = q.weights
getnodes(q::Quadrature)    = q.nodes
getnormals(q::Quadrature)  = q.normals
getelements(q::Quadrature) = q.elements
getbodies(q::Quadrature)   = q.bodies

Base.length(q::Quadrature) = length(getnodes(q))

function Base.append!(q::Quadrature,qa::Quadrature)
    el_new  = map(x-> x .+ length(q.nodes),qa.elements)
    bdy_new = map(x-> x .+ length(q.elements),qa.bodies)
    #NOTE: since the elements and bodies are indexed based on the local node
    # indexing, must shift the indexes, which is done using map below.
    append!(q.nodes,qa.nodes)
    append!(q.weights,qa.weights)
    append!(q.normals,qa.normals)
    append!(q.elements,el_new)
    append!(q.bodies,bdy_new)
    return q
end

function Base.empty!(q::Quadrature)
    empty!(q.nodes)
    empty!(q.weights)
    empty!(q.normals)
    empty!(q.elements)
    empty!(q.bodies)
    return q
end

function local_mesh_size(Y,iel,inode)
    nodes = getnodes(Y)
    idxs  = getelements(Y)[iel]
    hloc = mapreduce(min,idxs) do i
        if i==iel
            d = Inf
        else
            d = norm(nodes[iel]-nodes[i])
        end
    end
    return hloc
end

"""
    idx_nodes_to_elements(q::Quadrature)

For each node in `q`, return the indices of the elements to which it belongs.

Depending on the quadrature type, more efficient methods can be defined and overloaded if needed.
"""
function idx_nodes_to_elements(q)
    list = [Int[] for _ in 1:length(q)]
    for n in 1:length(getelements(q))
        for i in getelements(q)[n]
            push!(list[i],n)
        end
    end
    return list
end

"""
    idx_elements_to_bodies(q::Quadrature)

For each element in `q`, return the indices of the bodies to which it belongs.

Depending on the quadrature type, more efficient methods can be defined and overloaded if needed.
"""
function idx_elements_to_bodies(q::Quadrature)
    list = Vector{Int}(undef,length(getelements(q)))
    for n in 1:length(getbodies(q))
        for i in getbodies(q)[n]
            list[i] = n
        end
    end
    return list
end

"""
    idx_bodies_to_nodes(q::Quadrature)

For each body in `q`, return the indices of the nodes to belonging to it.

Depending on the quadrature type, more efficient methods can be defined and overloaded if needed.
"""
function idx_bodies_to_nodes(q::Quadrature)
    list = [Int[] for _ in 1:length(getbodies(q))]
    for n in 1:length(getbodies(q))
        bdy = getbodies(q)[n]
        els = getelements(q)[bdy]
        for i in els
            append!(list[n],i)
        end
    end
    return list
end

function _add_body_to_interaction_list!(list,X,Y)
    el2bdy   = idx_elements_to_bodies(Y)
    new_list = map(list) do l
        map(l) do (iel,inode)
            (el2bdy[iel],iel,inode)
        end
    end
    return new_list
end

"""
    _rescale_quadrature(p,origin,width,algo1d)

Convenience method to rescale the `p` points 1d quadrature rule `algo1d` from
`[-1,1]` to an interval given by `[origin,origin+width]`.

It is assume that `algo1d` is callable in the following: `nodes,weights =
algo1d(p)`, where `nodes` and `weights` are vectors.
"""
function _rescale_quadrature(p,origin,width,algo1d)
    nodes, weights = algo1d(p)
    @. nodes       = nodes*width/2            # shift to [-w/2,w/2]
    @. nodes       = nodes + width/2 + origin #shift to [a,b]
    @. weights     = weights * width/2
    return nodes, weights
end

function tensorquadrature(p::NTuple{N},el::Cuboid{N},algo) where {N}
    nodes1d   = Vector{Vector{Float64}}(undef,N)
    weights1d = Vector{Vector{Float64}}(undef,N)
    for n in 1:N
        nodes1d[n], weights1d[n] = _rescale_quadrature(p[n],el.low_corner[n],el.high_corner[n]-el.low_corner[n],algo)
    end
    nodes = Vector{Point{N,Float64}}(undef,prod(p))
    for (n,node) in enumerate(Iterators.product(nodes1d...))
        nodes[n] = Point(node)
    end
    weights = Vector{Float64}(undef,prod(p))
    for (n,weight) in enumerate(Iterators.product(weights1d...))
        weights[n] = prod(weight)
    end
    return Quadrature{N,Float64}(nodes,weights,[],[],[])
end

function tensorquadrature(p::NTuple{M},surf::ParametricSurface{M,N,T},algo) where {M,N,T}
    quad = Quadrature{N,T}()
    for element in  getelements(surf)
        # compute quadrature on reference element
        push!(quad.elements,[])
        ref_quad = tensorquadrature(p,element,algo) # quadrature in reference element
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

tensorquadrature(p::Int,bdy::ParametricBody{M},args...;kwargs...) where {M} = tensorquadrature(ntuple(i->p,M),bdy,args...;kwargs...)

function tensorquadrature(p::NTuple{M},bdy::ParametricBody{M,N,T},algo1d) where {M,N,T}
    quad = Quadrature{N,T}()
    for surf in bdy.patches
        for element in  getelements(surf)
            # compute quadrature on reference element
            push!(quad.elements,[])
            ref_quad = tensorquadrature(p,element,algo1d) # quadrature in reference element
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
    end
    push!(quad.bodies,1:length(quad.elements)|>collect)
    return quad
end

function tensorquadrature(p::NTuple{M},geo::Vector{ParametricBody{M,N,T}},algo) where {M,N,T}
    quad = Quadrature{N,T}()
    for bdy in geo
        push!(quad.bodies,[])
        for surf in bdy.patches
            for element in  getelements(surf)
                # compute quadrature on reference element
                push!(quad.elements,[])
                ref_quad = tensorquadrature(p,element,algo) # quadrature in reference element
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
                push!(quad.bodies[end],length(quad.elements))
            end
        end
    end
    return quad
end

"""
    GmshQuadrature

A singleton type representing a quadrature generated by `gmsh`.
"""
struct GmshQuadrature end

function gmshquadrature(qtype,dim,tag=-1)
    N = dim+1
    quad = Quadrature{N,Float64}()
    etypes, etags, ntags = gmsh.model.mesh.getElements(dim,tag)
    for i in 1:length(etypes)
        etype = etypes[i]
        nels  = length(etags[i])
        # get reference quadrature on element of type etype
        ipts, wstd = gmsh.model.mesh.getIntegrationPoints(etype,qtype)
        points_per_element = length(wstd)
        eproperty = gmsh.model.mesh.getElementProperties(etype)
        @info "Generating $qtype quadrature for elements of type $(eproperty[1])"
        jac, determ, pts = gmsh.model.mesh.getJacobians(etype,ipts,tag)
        jac = reshape(jac,3,3,:)
        pts = reshape(pts,3,:)
        normals = jac[:,N,:]
        # normalize
        for i=1:size(normals,2)
            normals[:,i] = normals[:,i]/norm(normals[:,i])
        end
        determ = reshape(determ,length(wstd),:)
        w      = vec(wstd.*determ)
        npts = length(w)
        for iel in 1:nels
            push!(quad.elements,[])
            for j in 1:points_per_element
                idx = (iel-1)*points_per_element + j
                push!(quad.nodes,pts[1:N,idx])
                push!(quad.normals,normals[1:N,idx])
                push!(quad.weights,w[idx])
                push!(quad.elements[end],length(quad.nodes))
            end
        end
    end
    return quad
end

################################################################################
## PLOT RECIPES
################################################################################
@recipe function f(quad::Quadrature{2})
    legend --> false
    grid   --> false
    aspect_ratio --> :equal
    nodes    = quad.nodes
    elements = quad.elements
    for n in 1:length(elements)
        el = elements[n]
        @series begin
            linecolor --> n
            pts = nodes[el]
            x = [pt[1] for pt in pts]
            y = [pt[2] for pt in pts]
            x,y
        end
    end
end

@recipe function f(quad::Quadrature{3})
    legend --> false
    grid   --> false
    # aspect_ratio --> :equal
    seriestype := :surface
    # color  --> :blue
    linecolor --> :black
    nodes    = quad.nodes
    elements = quad.elements
    for n in 1:length(elements)
        el = elements[n]
        @series begin
            fillcolor --> n
            pts = nodes[el]
            x = [pt[1] for pt in pts]
            y = [pt[2] for pt in pts]
            z = [pt[3] for pt in pts]
            x,y,z
        end
    end
end
