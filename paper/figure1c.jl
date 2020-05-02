using Nystrom, FastGaussQuadrature, Plots, LinearAlgebra
using Nystrom: error_interior_derivative_green_identity, error_exterior_derivative_green_identity

function fig_gen()
    dim = 2
    Γ = Domain(dim=dim)
    geo = circle()
    push!(Γ,geo)

    fig       = plot(yscale=:log10,xscale=:log10,xlabel="#dof",ylabel="error")
    qorder    = (2,3,4)
    h0        = 1.0
    niter     = 6
    operators = (Laplace(dim=2),Helmholtz(dim=2,k=2π))
    for op in operators
        # construct interior solution
        xout = (3,3)
        u    = (x)   -> SingleLayerKernel(op)(xout,x)
        dudn = (x,n) -> DoubleLayerKernel(op)(xout,x,n)
        for p in qorder
            conv_order = p - 1
            meshgen!(Γ,h0)
            ee_interior = []
            dof         = []
            for _ in 1:niter
                quadgen!(Γ,(p,),algo1d=gausslegendre)
                ADL  = AdjointDoubleLayerOperator(op,Γ)
                H    = HyperSingularOperator(op,Γ)
                δADL = GreensCorrection(ADL)
                δH = GreensCorrection(H)
                # test interior Green identity
                γ₀u       = γ₀(u,Γ)
                γ₁u       = γ₁(dudn,Γ)
                ee = error_interior_derivative_green_identity(ADL+δADL,H+δH,γ₀u,γ₁u)
                push!(ee_interior,norm(ee,Inf)/norm(γ₀u,Inf))
                push!(dof,length(Γ))
                @show length(Γ)
                refine!(Γ)
            end
            plot!(fig,dof,ee_interior,label=Nystrom.getname(op)*": p=$p",m=:o)
            plot!(fig,dof,1 ./(dof.^conv_order)*dof[end]^(conv_order)*ee_interior[end],
                  label="",linewidth=4)
        end
    end
    return fig
end

fig = fig_gen()
fname = "/home/lfaria/Dropbox/Luiz-Carlos/general_regularization/draft/figures/fig1c.pdf"
savefig(fig,fname)
display(fig)
