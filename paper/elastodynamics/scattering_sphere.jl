using Nystrom, FastGaussQuadrature, Plots, LinearAlgebra, IterativeSolvers, ParametricSurfaces, LinearMaps, LaTeXStrings
using Nystrom: Point
using StaticArrays, SparseArrays
using gmsh

function pwave_exterior_neumann3d()
    λ,μ,ρ,ω = 2,1,1,10π
    pde = Elastodynamic(dim=3,μ=μ,λ=λ,ρ=ρ,ω=ω)
    # construct exterior solution
    p       = 4
    niter   = 1
    c       = ones(3)
    kp      = pde.ω*sqrt(pde.ρ)/sqrt(λ+2*μ)
    θ       = π/2
    ϕ       = 0
    d       = SVector{3,Float64}(sin(θ)*cos(ϕ), sin(θ)*sin(ϕ), cos(θ))
    ui      = (x)   -> d*exp(im*kp*dot(x,d))
    ti      = (x,n) -> im*kp*exp(im*kp*dot(x,d)) * (λ*I + 2*μ*d*d')*n
    # far field
    gmsh.initialize()
    gmsh.clear()
    # geo = gmsh.model.occ.addTorus(0,0,0,1,0.5)
    geo = gmsh.model.occ.addSphere(0,0,0,1)
    gmsh.model.occ.synchronize()
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax",0.14)
    gmsh.model.mesh.generate(2)
    gmsh.model.mesh.setOrder(3)
    gmsh.write("torus.msh")
    eFar  = []
    dof = []
    Γ      = quadgengmsh("Gauss$p",((2,geo),))
    @info length(Γ), sum(Γ.weights) - 4π
    γ₀u       = γ₀(ui,Γ)
    γ₁u       = γ₁(ti,Γ)
    # S,D    = single_double_layer(pde,Γ)
    # S*γ₁u - D*γ₀u - γ₀u/2
    ADL,H  = adjointdoublelayer_hypersingular(pde,Γ)
    # solve exterior problem
    η         = im*pde.ω
    L         = LinearMap(3*length(γ₀u)) do x
        T = eltype(x)
        σ = reinterpret(SVector{3,T},x)
        reinterpret(T,η*σ/2 - η*(ADL*σ) + H*σ )
    end
    σ,ch      = gmres(L,reinterpret(ComplexF64,-γ₁u),verbose=false,log=true,tol=1e-4,restart=200,maxiter=200)
    σ         = reinterpret(SVector{3,ComplexF64},σ)
    @show length(Γ), ch.iters
    S,D = nothing, nothing
    Base.GC.gc() # clean things up
    # output
    gmsh.clear()
    h = 2π/(10*kp)
    lx = -4; wx = 8
    ly = -4; wy = 8
    rec = gmsh.model.occ.addRectangle(lx,ly,0,wx,wy)
    gmsh.model.occ.synchronize()
    gmsh.model.mesh.setTransfiniteSurface(rec)
    gmsh.model.mesh.setRecombine(2,rec)
    Nx,Ny = ceil(Int,wx/h), ceil(Int,wy/h)
    gmsh.model.mesh.setTransfiniteCurve(1,Nx)
    gmsh.model.mesh.setTransfiniteCurve(2,Ny)
    gmsh.model.mesh.setTransfiniteCurve(3,Nx)
    gmsh.model.mesh.setTransfiniteCurve(4,Ny)
    gmsh.model.mesh.generate(2)
    gmsh.write("xyplane.msh")
    output_tags,output_coords,_ = gmsh.model.mesh.getNodes()
    output_pts                  = reinterpret(Point{3,Float64},output_coords) |> collect |> vec
    D = DoubleLayerOperator(pde,output_pts,Γ)
    S = SingleLayerOperator(pde,output_pts,Γ)
    # near field correction
    δS = GreensCorrection(S,S,D)
    δD = GreensCorrection(D,δS.R,δS.L,δS.idxel_near)  # reuse precomputed quantities of δS
    δSsp = sparse(δS)
    δDsp = sparse(δD)
    Ui        = [ui(x) for x in output_pts]
    Us        = D*σ + δDsp*σ - η*(S*σ + δSsp*σ)
    Ut        = Us .+ Ui # total field
    sol_real = vec([ Vector(real(Ut[n])) for n in 1:length(Ut)])
    sol_imag = vec([ Vector(imag(Ut[n])) for n in 1:length(Ut)])
    view_solution = gmsh.view.add("solution")
    mname = gmsh.model.getCurrent()
    gmsh.view.addModelData(view_solution,0,mname,"NodeData",output_tags,sol_real)
    gmsh.view.addModelData(view_solution,1,mname,"NodeData",output_tags,sol_imag)
    gmsh.view.write(view_solution, "output_sphere.pos")
    gmsh.finalize()
    return ch
end

ch = pwave_exterior_neumann3d()
# fname     = "/home/lfaria/Dropbox/Luiz-Carlos/general_regularization/draft/figures/elastody_neum_er.pdf"
# savefig(fig1,fname)
# fname     = "/home/lfaria/Dropbox/Luiz-Carlos/general_regularization/draft/figures/elastody_neum_iters.pdf"
# savefig(fig2,fname)