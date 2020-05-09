using Nystrom, FastGaussQuadrature, Plots, LinearAlgebra
using Nystrom: error_interior_green_identity, error_exterior_green_identity, sphere_sources

function fig_gen()
    dim = 3
    Γ = Domain(dim=dim)
    geo = sphere()
    R   = 10
    xtest = sphere_sources(nsources=100;radius=R)
    push!(Γ,geo)

    fig       = plot(yscale=:log10,xscale=:log10,xlabel="√N",ylabel="error")
    qorder    = (2,)
    h0        = 2
    niter     = 5
    k         = 2π
    operators = (Helmholtz(dim=dim,k=k),)
    for op in operators
        # construct exterior solution
        xin = (0.2,0.1,-0.1)
        u    = (x)   -> SingleLayerKernel(op)(xin,x)
        dudn = (x,n) -> DoubleLayerKernel(op)(xin,x,n)
        for p in qorder
            conv_order = p + 1
            meshgen!(Γ,h0)
            ee  = []
            dof = []
            for _ in 1:niter
                quadgen!(Γ,p;algo1d=gausslegendre)
                S,D = single_double_layer(op,Γ)
                # test exterior Green identity
                γ₀u       = γ₀(u,Γ)
                γ₁u       = γ₁(dudn,Γ)
                η = im*k
                σ,ch  = gmres(I/2 + D - η*S,γ₀u,verbose=false,log=true,tol=1e-14,restart=100)
                ua(x) = DoubleLayerPotential(op,Γ)(σ,x) - η*SingleLayerPotential(op,Γ)(σ,x)
                Ue    = [u(x) for x in xtest]
                Ua    = [ua(x) for x in xtest]
                push!(ee,norm(Ue.-Ua,Inf)/norm(Ue,Inf))
                @show length(Γ),ee[end]
                push!(dof,length(Γ))
                refine!(Γ)
            end
            dof = sqrt.(dof)
            plot!(fig,dof,ee,label=Nystrom.getname(op)*": p=$p",m=:o,color=conv_order)
            plot!(fig,dof,1 ./(dof.^conv_order)*dof[end]^(conv_order)*ee[end],
                  label="",linewidth=4,line=:dot,color=conv_order)
        end
    end
    return fig
end

fig = fig_gen()
fname = "/home/lfaria/Dropbox/Luiz-Carlos/general_regularization/draft/figures/fig3b.pdf"
savefig(fig,fname)
display(fig)
