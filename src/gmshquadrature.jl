struct Quadrature{N,T}
    nodes::Vector{Point{N,T}}
    weights::Vector{T}
    normals::Vector{Normal{N,T}}
    elements::Vector{Vector{Int}}
end
Quadrature{N,T}() where {N,T} = Quadrature{N,T}([],[],[],[])

Base.length(q::Quadrature) = length(q.weights)
getnodes(q::Quadrature)    = q.nodes
getnormals(q::Quadrature)  = q.normals
getweights(q::Quadrature)  = q.weights
getelements(q::Quadrature) = q.elements
getnodes(q::Quadrature,I)   = q.nodes[I]
getnormals(q::Quadrature,I) = q.normals[I]
getweights(q::Quadrature,I) = q.weights[I]
getelements(q::Quadrature,I)= q.elements[I]

function Base.permute!(quad::Quadrature,perm::Vector{Int})
    map(x->permute!(x,perm),(quad.nodes,quad.normals,quad.weights))
    iperm = invperm(perm)
    for el in quad.elements
        setindex!(el,iperm[el],:)
    end
    return quad
end

function quadgengmsh(qtype,dimtags)
    N = 3
    quad = Quadrature{N,Float64}()
    for (dim,tag) in dimtags
        etypes, etags, ntags = gmsh.model.mesh.getElements(dim,tag)
        @info "Using gmsh to generate a quadrature..."
        for i in 1:length(etypes)
            etype = etypes[i]
            nels  = length(etags[i])
            # get reference quadrature on element of type etype
            ipts, wstd = gmsh.model.mesh.getIntegrationPoints(etype,qtype)
            points_per_element = length(wstd)
            eproperty = gmsh.model.mesh.getElementProperties(etype)
            @info "\t Generating $qtype quadrature for elements of type $(eproperty[1]).
                   Quadrature contains $(points_per_element) point per element."
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
    end
    @info "done."
    return quad
end

# function quadgen(qtype,dim,tag=-1)
#     etypes, etags, ntags = gmsh.model.mesh.getElements(dim,tag)
#     N = dim+1
#     quads = Vector{Quadrature{N,Float64}}()
#     for etype in etypes
#         # get reference quadrature on element of type etype
#         ipts, wstd = gmsh.model.mesh.getIntegrationPoints(etype,qtype)
#         eproperty = gmsh.model.mesh.getElementProperties(etype)
#         @info "Generating $qtype quadrature for elements of type $(eproperty[1])"
#         jac, determ, pts = gmsh.model.mesh.getJacobians(etype,ipts,tag)
#         jac = reshape(jac,3,3,:)
#         pts = reshape(pts,3,:)
#         normals = jac[:,N,:]
#         for i=1:size(normals,2); normals[:,i] = normals[:,i]/norm(normals[:,i]); end
#         determ = reshape(determ,length(wstd),:)
#         w      = vec(wstd.*determ)
#         npts = length(w)
#         quad = Quadrature{dim+1,Float64}([],[],[],length(wstd))
#         for i in 1:npts
#             push!(quad.nodes,pts[1:N,i])
#             push!(quad.normals,normals[1:N,i])
#             push!(quad.weights,w[i])
#         end
#         push!(quads,quad)
#     end
#     return quads
# end
