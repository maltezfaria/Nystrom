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
    dof = []
    niters = []
    h   = 10 .^(range(-0.85,stop=0,length=10))
    gmsh.option.setNumber("Mesh.ElementOrder",3)
    for n in 1:length(h)
        # create a sphere in gmsh
        gmsh.clear()
        geo = gmsh.model.occ.addSphere(0,0,0,1)
        gmsh.model.occ.synchronize()
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin",h[n])
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax",h[n])
        gmsh.model.mesh.generate(2)
        Γ     = quadgengmsh("Gauss$p",((2,geo),))
        @show length(Γ)
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
        if n==1
            global uref = deepcopy(Ut)
        else
            push!(eFar,norm(Ut-uref,Inf)/norm(uref,Inf))
            push!(dof,3*length(Γ))
        end
    end
    gmsh.finalize()
    return h,dof,eFar,niters
end

h,dof,eFar,niters = pwave_exterior_neumann3d()
using JLD
save("convergence_sphere.jld","h",h,"dof",dof,"eFar",eFar,"niters",niters)
