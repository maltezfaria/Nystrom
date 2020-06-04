using Nystrom, FastGaussQuadrature, Plots, LinearAlgebra, IterativeSolvers, ParametricSurfaces, LinearMaps, LaTeXStrings
using Nystrom: Point
using StaticArrays
using gmsh

function pwave_exterior_neumann3d()
    λ,μ,ρ,ω = 2,1,1,π
    pde = Elastodynamic(dim=3,μ=μ,λ=λ,ρ=ρ,ω=ω)
    p       = 4
    # construct incident wave
    kp      = pde.ω*sqrt(pde.ρ)/sqrt(λ+2*μ)
    θ       = π/2
    ϕ       = 0
    d       = SVector{3,Float64}(sin(θ)*cos(ϕ), sin(θ)*sin(ϕ), cos(θ))
    ui      = (x)   -> d*exp(im*kp*dot(x,d))
    ti      = (x,n) -> im*kp*exp(im*kp*dot(x,d)) * (λ*I + 2*μ*d*d')*n
    # create output mesh
    gmsh.initialize()
    gmsh.clear()
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax",0.4)
    sph = gmsh.model.occ.addSphere(0,0,0,5)
    gmsh.model.occ.synchronize()
    gmsh.model.mesh.generate(2)
    output_tags,output_coords,_ = gmsh.model.mesh.getNodes()
    output_pts                  = reinterpret(Point{3,Float64},output_coords) |> collect |> vec
    # begin iterations
    cc = 0
    eFar  = []
    sol   = []
    dof = []
    niters = []
    gmsh.option.setNumber("Mesh.ElementOrder",3)
    gmsh.clear()
    geo = gmsh.model.occ.addSphere(0,0,0,1)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin",2)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax",2)
    gmsh.model.occ.synchronize()
    gmsh.model.mesh.generate(2)
    for n in 1:4
        # create a sphere in gmsh
        Γ     = quadgengmsh("Gauss$p",((2,geo),))
        γ₀u   = γ₀(ui,Γ)
        γ₁u   = γ₁(ti,Γ)
        ADL,H = adjointdoublelayer_hypersingular(pde,Γ)
        # solve exterior problem
        η = im*pde.ω
        L = LinearMap(3*length(γ₀u)) do x
            T = eltype(x)
            σ = reinterpret(SVector{3,T},x)
            reinterpret(T,η*σ/2 - η*(ADL*σ) + H*σ )
        end
        σ,ch      = gmres(L,reinterpret(ComplexF64,-γ₁u),verbose=false,log=true,tol=1e-8,restart=200,maxiter=200)
        σ         = reinterpret(SVector{3,ComplexF64},σ)
        ADL,H     = nothing, nothing
        Base.GC.gc() # force garbage collector to pass and clear ADL,H
        @show length(Γ), ch.iters
        push!(niters,ch.iters)
        # compute error on reference surface
        us(x)     = DoubleLayerPotential(pde,Γ)(σ,x) - η*SingleLayerPotential(pde,Γ)(σ,x)
        Ui        = [ui(x) for x in output_pts]
        Us        = [us(x) for x in output_pts]
        Ut        = Us .+ Ui
        push!(sol,Ut)
        push!(dof,3*length(Γ))
        gmsh.model.mesh.refine()
        gmsh.model.mesh.setOrder(3) # this has to be done after refine for some reason
    end
    gmsh.finalize()
    return dof,sol,niters
end

dof,sol,niters = pwave_exterior_neumann3d()

exact = sol[end]
eFar = []
for n=1:length(sol)-1
    push!(eFar,norm(exact-sol[n],Inf)/norm(exact,Inf))
end

using JLD
save("convergence_sphere2.jld","dof",dof[1:end-1],"eFar",eFar,"niters",niters[1:end-1])
