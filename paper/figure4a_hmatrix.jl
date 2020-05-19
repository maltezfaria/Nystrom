using Nystrom, FastGaussQuadrature, Plots, LinearAlgebra, IterativeSolvers, ParametricSurfaces, HMatrices, Clusters
using Nystrom: circle_helmholtz_soundsoft, error_exterior_green_identity, Point

function scattering_helmholtz_circle_soundsoft(qorder,h,niter)
    dim = 2
    R = 1
    geo   = Circle(radius=R)
    xtest = [Point(2*R*cos(θ),2*R*sin(θ)) for θ in 0:0.1:2π]
    fig       = plot(yscale=:log10,xscale=:log10,xlabel="N",ylabel="error")
    k         = 8π
    pde       = Helmholtz(dim=dim,k=k)
    ue(x) = circle_helmholtz_soundsoft(x;radius=R,θin=θ,k=k)
    θ     = 2π/3
    kx    = cos(θ)*k
    ky    = sin(θ)*k
    for p in qorder
        conv_order = p + 1
        meshgen!(geo,h)
        gmres_iter = []
        dof        = []
        ee         = []
        for _ in 1:niter
            Γ = quadgen(geo,p,gausslegendre)
            spl = GeometricMinimalSplitter(nmax=10)
            clt = ClusterTree(Γ.nodes,spl)
            bclt = BlockTree(clt,clt)
            permute!(Γ,clt.perm)
            Sop, Dop = SingleLayerOperator(pde,Γ), DoubleLayerOperator(pde,Γ)
            atol = 1e-10
            compression = HMatrices.PartialACA(atol=atol)
            S_matrix = HMatrix(Sop,bclt,compression)
            D_matrix = HMatrix(Dop,bclt,compression)
            δS = GreensCorrection(Sop,S_matrix,D_matrix)
            δD = GreensCorrection(Dop,S_matrix,D_matrix)
            S  = S_matrix + sparse(δS)
            D  = D_matrix + sparse(δD)
            # S,D   = single_double_layer(pde,Γ)
            γ₀U   = γ₀((y) -> -exp(im*(kx*y[1] + ky*y[2])),Γ)
            σ,ch  = gmres(I/2 + D - im*k*S,γ₀U,verbose=false,log=true,tol=10*atol,restart=100,maxiter=100)
            ua(x) = DoubleLayerPotential(pde,Γ)(σ,x) - im*k*SingleLayerPotential(pde,Γ)(σ,x)
            Ue    = [ue(x) for x in xtest]
            Ua    = [ua(x) for x in xtest]
            push!(ee,norm(Ue.-Ua,Inf)/norm(Ue,Inf))
            @show length(Γ),ee[end], ch.iters
            push!(dof,length(Γ))
            refine!(geo)
        end
        plot!(fig,dof,ee,label=Nystrom.getname(pde)*": p=$p",m=:o,color=p)
        plot!(fig,dof,1 ./(dof.^conv_order)*dof[end]^(conv_order)*ee[end],
              label="",linewidth=4,color=p)
    end
    return fig
end


qorder = (2,3,4)
h      = 0.1
niter  = 6
fig    = scattering_helmholtz_circle_soundsoft(qorder,h,niter)
fname = "/home/lfaria/Dropbox/Luiz-Carlos/general_regularization/draft/figures/fig4a_hmatrix.pdf"
savefig(fig,fname)
display(fig)

