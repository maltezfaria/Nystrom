struct Domain{N,T}
    bodies::Vector
    quadrature::Quadrature{<:Any,N,T}
end
Domain{N,T}() where {N,T} = Domain{N,T}([],Quadrature{Any,N,T}())
Domain{N}(args...) where {N} = Domain{N,Float64}(args...)
Domain(;dim) = Domain{dim}()

ambient_dim(Γ::Domain{N}) where {N} = N

Base.push!(Γ::Domain,bdy::ParametricBody) = push!(Γ.bodies,bdy)

function quadgen!(Γ::Domain{N,T},p,algo1d) where {N,T}
    empty!(Γ.quadrature)
    for bdy in getbodies(Γ)
        if bdy isa ParametricBody
            qbdy = tensorquadrature(p,bdy,algo1d)
            append!(Γ.quadrature,qbdy)
        # elseif bdy ias GmshBody
        #     qbdy = quadgen!(bdy,h,p)
        #     append!(quad,qbdy)
        end
    end
    return Γ
end

function meshgen!(Γ::Domain,h)
    for bdy in getbodies(Γ)
        meshgen!(bdy,h)
    end
    return Γ
end

function refine!(Γ::Domain)
    for bdy in getbodies(Γ)
        refine!(bdy)
    end
    return Γ
end

getnodes(Γ::Domain)    = getnodes(Γ.quadrature)
getweights(Γ::Domain)  = getweights(Γ.quadrature)
getnormals(Γ::Domain)  = getnormals(Γ.quadrature)
getelements(Γ::Domain) = getelements(Γ.quadrature)
getbodies(Γ::Domain)   = Γ.bodies

Base.length(Γ::Domain) = length(Γ.quadrature)

################################################################################
## PLOT RECIPES
################################################################################
@recipe function f(Γ::Domain)
    Γ.quadrature
end
