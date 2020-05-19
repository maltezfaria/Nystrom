using Nystrom, FastGaussQuadrature, Plots, LinearAlgebra, IterativeSolvers, ParametricSurfaces
using Nystrom: circle_helmholtz_soundsoft, error_exterior_green_identity, Point

function scattering_helmholtz_circle_soundsoft(qorder,h,niter)
    dim = 2
    R = 1
    geo = Circle(radius=R)
    xtest = [Point(2*R*cos(θ),2*R*sin(θ)) for θ in 0:0.1:2π]
    fig       = plot(yscale=:log10,xscale=:log10,xlabel="N",ylabel="error")
    k         = 2π
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
            S,D   = single_double_layer(pde,Γ)
            γ₀U   = γ₀((y) -> -exp(im*((kx,ky) ⋅ y)),Γ)
            σ,ch  = gmres(I/2 + D - im*k*S,γ₀U,verbose=false,log=true,tol=1e-14,restart=100)
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
niter  = 4
fig    = scattering_helmholtz_circle_soundsoft(qorder,h,niter)
fname = "/home/lfaria/Dropbox/Luiz-Carlos/general_regularization/draft/figures/fig4a.pdf"
savefig(fig,fname)
display(fig)

