struct GsmhBody
    dim::Int
    tags::Vector{Int}
end

function gmshquadrature(qtype,bdy::GmshBody)
    dim  = bdy.dim
    tags = bdy.tags
    quad = Quadrature{3,Float64}()
    for tag in tags
        _quad = gmshquadrature(qtype,dim,tag)
        quad  = append!(quad,_quad)
    end
    return quad
end

function gmshquadrature(qtype,dim,tag=-1)
    N = 3
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
