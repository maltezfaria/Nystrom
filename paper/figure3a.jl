using Nystrom, FastGaussQuadrature, Plots, LinearAlgebra
using Nystrom: error_interior_green_identity, error_exterior_green_identity

function fig_gen()
    dim = 2
    Γ = Domain(dim=dim)
    geo = kite()
    R   = 5
    xtest = [Point(R*cos(θ),R*sin(θ)) for θ in 0:0.1:2π]
    push!(Γ,geo)

    fig       = plot(yscale=:log10,xscale=:log10,xlabel="N",ylabel="error")
    qorder    = (2,3,4)
    h0        = 0.5
    niter     = 9
    k = 2π
    operators = (Helmholtz(dim=dim,k=k),)
    for op in operators
        # construct exterior solution
        xin = (0.2,0.1)
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
                # test interior Green identity
                γ₀u       = γ₀(u,Γ)
                γ₁u       = γ₁(dudn,Γ)
                σ,ch  = gmres(I/2 + D - im*k*S,γ₀u,verbose=false,log=true,tol=1e-14,restart=100)
                ua(x) = DoubleLayerPotential(op,Γ)(σ,x) - im*k*SingleLayerPotential(op,Γ)(σ,x)
                Ue    = [u(x) for x in xtest]
                Ua    = [ua(x) for x in xtest]
                push!(ee,norm(Ue.-Ua,Inf)/norm(Ue,Inf))
                @show length(Γ),ee[end]
                push!(dof,length(Γ))
                refine!(Γ)
            end
            plot!(fig,dof,ee,label=Nystrom.getname(op)*": p=$p",m=:o,color=conv_order)
            plot!(fig,dof,1 ./(dof.^conv_order)*dof[end]^(conv_order)*ee[end],
                  label="",linewidth=4,line=:dot,color=conv_order)
        end
    end
    return fig
end

fig = fig_gen()
fname = "/home/lfaria/Dropbox/Luiz-Carlos/general_regularization/draft/figures/fig3a.pdf"
savefig(fig,fname)
display(fig)
