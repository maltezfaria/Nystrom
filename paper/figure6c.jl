using Nystrom, FastGaussQuadrature, Plots, LinearAlgebra, IterativeSolvers, GmshTools
using Nystrom: circle_helmholtz_soundsoft, error_exterior_green_identity, Point

function fig_gen()
    dim = 2
    Γ = Domain(dim=dim)
    R = 1
    gmsh.initialize()
    gmsh.clear()
    geo = gmsh.model.occ.addCircle(0,0,0,R)
    gmsh.model.occ.synchronize()
    gmsh.model.mesh.generate(1)
    quad = gmshquadrature("Gauss4",1,geo)
    @test sum(quad.weights) - 2π*r < 1e-1 # a rough calculation
    gmsh.finalize()
    geo = circle(radius=R)
    xtest = [Point(2*R*cos(θ),2*R*sin(θ)) for θ in 0:0.1:2π]
    push!(Γ,geo)
    fig       = plot(yscale=:log10,xscale=:log10,xlabel="#dof",ylabel="error")

    qorder    = (2,3,4)
    h0        = 1.0
    niter     = 8
    k         = 2π
    pde       = Helmholtz(dim=dim,k=k)
    ue(x) = circle_helmholtz_soundsoft(x;radius=R,θin=θ,k=k)
    θ     = 2π*rand()
    kx    = cos(θ)*k
    ky    = sin(θ)*k
    for p in qorder
        conv_order = p + 1
        meshgen!(Γ,h0)
        gmres_iter = []
        dof        = []
        ee         = []
        for _ in 1:niter
            quadgen!(Γ,p;algo1d=gausslegendre)
            S,D   = single_double_layer(pde,Γ)
            γ₀U   = [-exp(im*(kx*y[1] + ky*y[2]))  for  y in getnodes(Γ)]
            σ,ch  = gmres(I/2 + D - im*k*S,γ₀U,verbose=true,log=true,tol=1e-14,restart=100)
            ua(x) = DoubleLayerPotential(pde,Γ)(σ,x) - im*k*SingleLayerPotential(pde,Γ)(σ,x)
            Ue    = [ue(x) for x in xtest]
            Ua    = [ua(x) for x in xtest]
            push!(ee,norm(Ue.-Ua,Inf)/norm(Ue,Inf))
            @show length(Γ),ee[end]
            push!(dof,length(Γ))
            refine!(Γ)
        end
        plot!(fig,dof,ee,label=Nystrom.getname(pde)*": p=$p",m=:o,color=p)
        plot!(fig,dof,1 ./(dof.^conv_order)*dof[end]^(conv_order)*ee[end],
              label="",linewidth=4,color=p)
    end
    return fig
end

fig = fig_gen()
fname = "/home/lfaria/Dropbox/Luiz-Carlos/general_regularization/draft/figures/fig6a.pdf"
savefig(fig,fname)
display(fig)

