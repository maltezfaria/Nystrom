using Nystrom, FastGaussQuadrature, Plots, LinearAlgebra, IterativeSolvers, Clusters, HMatrices, SparseArrays, LinearMaps, GmshTools
using Nystrom: sphere_helmholtz_soundsoft, error_exterior_green_identity, Point, @gmsh

function fig_gen()
    dim = 3
    R = 1
    qorder    = (1,)
    # h0        = 0.25

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.option.setNumber("Mesh.Algorithm",8)
    fig       = plot(yscale=:log10,xscale=:log10,xlabel="√N",ylabel="error")
    niter     = 3
    k         = 2π
    pde       = Helmholtz(dim=dim,k=k)
    θ         = rand()*π
    ϕ         = rand()*2π
    ue(x) = sphere_helmholtz_soundsoft(x;radius=R,k=k,θin=θ,ϕin = ϕ)
    xtest = [Point(2*R*sin(θ)*cos(ϕ),2*R*sin(θ)*sin(ϕ),2*R*cos(θ)) for θ in 0:0.1:π, ϕ in 0:0.1:2π]
    Ue    = [ue(x) for x in xtest]
    kx = k*sin(θ)*cos(ϕ)
    ky = k*sin(θ)*sin(ϕ)
    kz = k*cos(θ)
    for p in qorder
        gmsh.clear()
        geo = gmsh.model.occ.addSphere(0,0,0,1)
        gmsh.model.occ.synchronize()
        # Nystrom.gmsh_set_meshsize(h0)
        gmsh.model.mesh.generate(2)
        gmsh.model.mesh.recombine()
        conv_order = p + 1
        gmres_iter = []
        dof        = []
        ee         = []
        for _ in 1:niter
            gmsh.model.mesh.setOrder(conv_order+1) # set it higher than the method's error to avoid geometrical errors
            Γ     = quadgengmsh("Gauss$p",2,-1)
            push!(dof,length(Γ))
            @info "starting computation with $(length(Γ.nodes)) quadrature points..."
            spl = CardinalitySplitter(nmax=128)
            clt = ClusterTree(Γ.nodes,spl)
            bclt = BlockTree(clt,clt)
            permute!(Γ,clt.perm)
            atol = 1e-5
            compress = (x) -> HMatrix(x,bclt,HMatrices.PartialACA(atol=atol))
            @info "Assembling matrices..."
            S,D   = single_double_layer(pde,Γ;compress=compress)
            @info "\t compression rate: S --> $(HMatrices.compression_rate(S)), D --> $(HMatrices.compression_rate(D)),"
            @info "done..."
            @info "solving by gmres..."
            γ₀U   = γ₀((y) -> -exp(im*((kx,ky,kz) ⋅ y)),Γ)
            L = LinearMap(σ -> σ/2 + D*σ - im*k*(S*σ),length(γ₀U))
            σ,ch  = gmres(L,γ₀U,verbose=false,log=true,tol=1e-6,restart=100,maxiter=100)
            @info "done."
            @info "Evaluating field error..."
            ua(x) = DoubleLayerPotential(pde,Γ)(σ,x) - im*k*SingleLayerPotential(pde,Γ)(σ,x)
            Ua    = [ua(x) for x in xtest]
            push!(ee,norm(Ue.-Ua,Inf)/norm(Ue,Inf))
            @info "gmres converged in $(ch.iters) iteratios. Field error was $(ee[end]) "
            gmsh.model.mesh.refine()
        end
        dof = sqrt.(dof) # dof per dimension, roughly inverse of mesh size
        plot!(fig,dof,ee,m=:o,color=p,label="")
        plot!(fig,dof,1 ./(dof.^conv_order)*dof[end]^(conv_order)*ee[end],
              linewidth=4,color=p,label=Nystrom.getname(pde)*": slope $(conv_order)",)
    end
    gmsh.finalize()
    return fig
end

fig = fig_gen()
fname = "/home/lfaria/Dropbox/Luiz-Carlos/general_regularization/draft/figures/hybrid_mesh_convergence.pdf"
savefig(fig,fname)
display(fig)
