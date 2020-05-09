using Nystrom, FastGaussQuadrature, Plots, LinearAlgebra, IterativeSolvers
using Nystrom: sphere_helmholtz_soundsoft, error_exterior_green_identity, Point

function fig_gen()
    dim = 3
    Γ = Domain(dim=dim)
    R = 1
    geo = sphere(radius=R)
    push!(Γ,geo)
    fig       = plot(yscale=:log10,xscale=:log10,xlabel="sqrt(N)",ylabel="error")

    qorder    = (2,3,4)
    h0        = 2.0
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
        conv_order = p + 1
        meshgen!(Γ,h0)
        gmres_iter = []
        dof        = []
        ee         = []
        for _ in 1:niter
            quadgen!(Γ,p;algo1d=gausslegendre)
            S,D   = single_double_layer(pde,Γ)
            γ₀U   = [-exp(im*(kx*y[1] + ky*y[2] +  kz*y[3]))  for  y in getnodes(Γ)]
            σ,ch  = gmres(I/2 + D - im*k*S,γ₀U,verbose=false,log=true,tol=1e-14,restart=100)
            ua(x) = DoubleLayerPotential(pde,Γ)(σ,x) - im*k*SingleLayerPotential(pde,Γ)(σ,x)
            Ua    = [ua(x) for x in xtest]
            push!(ee,norm(Ue.-Ua,Inf)/norm(Ue,Inf))
            push!(dof,length(Γ))
            @show length(Γ), ee[end]
            refine!(Γ)
        end
        dof = sqrt.(dof) # dof per dimension, roughly inverse of mesh size
        plot!(fig,dof,ee,label=Nystrom.getname(pde)*": p=$p",m=:o,color=p)
        plot!(fig,dof,1 ./(dof.^conv_order)*dof[end]^(conv_order)*ee[end],
              label="",linewidth=4,color=p)
    end
    return fig
end

fig = fig_gen()
fname = "/home/lfaria/Dropbox/Luiz-Carlos/general_regularization/draft/figures/fig6b.pdf"
savefig(fig,fname)
display(fig)

