using Nystrom, FastGaussQuadrature, Plots, LinearAlgebra, IterativeSolvers, ParametricSurfaces, Clusters, HMatrices, SparseArrays, LinearMaps
using Nystrom: sphere_helmholtz_soundsoft, error_exterior_green_identity, Point

function fig_gen()
    dim = 3
    R = 1
    geo = Sphere(radius=R)
    fig       = plot(yscale=:log10,xscale=:log10,xlabel="√N",ylabel="error")
    qorder    = (2,3,4)
    h0        = 2.0
    niter     = 5
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
        conv_order = p + 1
        meshgen!(geo,h0)
        gmres_iter = []
        dof        = []
        ee         = []
        for _ in 1:niter
            Γ     = quadgen(geo,p,gausslegendre)
            spl = GeometricMinimalSplitter(nmax=10)
            clt = ClusterTree(Γ.nodes,spl)
            bclt = BlockTree(clt,clt)
            permute!(Γ,clt.perm)
            Sop, Dop = SingleLayerOperator(pde,Γ), DoubleLayerOperator(pde,Γ)
            atol = 1e-10
            compress = (x) -> HMatrix(x,bclt,HMatrices.PartialACA(atol=atol))
            S,D   = single_double_layer(pde,Γ;compress=compress)
            γ₀U   = γ₀((y) -> -exp(im*((kx,ky,kz) ⋅ y)),Γ)
            L = LinearMap(σ -> σ/2 + D*σ - im*k*S*σ,length(γ₀U))
            σ,ch  = gmres(L,γ₀U,verbose=false,log=true,tol=100*atol,restart=100,maxiter=100)
            ua(x) = DoubleLayerPotential(pde,Γ)(σ,x) - im*k*SingleLayerPotential(pde,Γ)(σ,x)
            Ua    = [ua(x) for x in xtest]
            push!(ee,norm(Ue.-Ua,Inf)/norm(Ue,Inf))
            push!(dof,length(Γ))
            @show length(Γ), ee[end], ch.iters
            refine!(geo)
        end
        dof = sqrt.(dof) # dof per dimension, roughly inverse of mesh size
        plot!(fig,dof,ee,label=Nystrom.getname(pde)*": p=$p",m=:o,color=p)
        plot!(fig,dof,1 ./(dof.^conv_order)*dof[end]^(conv_order)*ee[end],
              label="",linewidth=4,color=p)
    end
    return fig
end

fig = fig_gen()
fname = "/home/lfaria/Dropbox/Luiz-Carlos/general_regularization/draft/figures/fig4b_hmat.pdf"
savefig(fig,fname)
display(fig)

