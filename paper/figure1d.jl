using Nystrom, FastGaussQuadrature, Plots, LinearAlgebra, LaTeXStrings
using Nystrom: error_interior_derivative_green_identity, error_exterior_derivative_green_identity

function fig_gen()
    dim = 2
    Γ = Domain(dim=dim)
    geo = circle()
    push!(Γ,geo)

    fig       = plot(yscale=:log10,xscale=:log10,xlabel="N",ylabel="error")
    qorder    = (2,3,4)
    h0        = 1.0
    niter     = 6
    operators = (Elastodynamic(dim=2,λ=1,μ=1,ρ=1,ω=1),)
    for op in operators
        # construct interior solution
        c    = rand(dim)
        xout = (3,3)
        u    = (x)   -> SingleLayerKernel(op)(xout,x)*c
        dudn = (x,n) -> transpose(DoubleLayerKernel(op)(xout,x,n))*c
        for p in qorder
            conv_order = p - 1
            meshgen!(Γ,h0)
            ee_interior = []
            dof         = []
            for _ in 1:niter
                quadgen!(Γ,(p,),algo1d=gausslegendre)
                ADL,H = adjointdoublelayer_hypersingular(op,Γ)
                # test interior Green identity
                γ₀u  = γ₀(u,Γ)
                γ₁u  = γ₁(dudn,Γ)
                ee   = error_interior_derivative_green_identity(ADL,H,γ₀u,γ₁u)
                push!(ee_interior,norm(ee,Inf)/norm(γ₀u,Inf))
                push!(dof,length(Γ))
                refine!(Γ)
            end
            plot!(fig,dof,ee_interior,label=Nystrom.getname(op)*": p=$p",m=:o,color=conv_order)
            plot!(fig,dof,1 ./(dof.^conv_order)*dof[end]^(conv_order)*ee_interior[end],
                  label="",linewidth=4,line=:dot,color=conv_order)
        end
    end
    return fig
end

fig = fig_gen()
fname = "/home/lfaria/Dropbox/Luiz-Carlos/general_regularization/draft/figures/fig1d.pdf"
savefig(fig,fname)
display(fig)
