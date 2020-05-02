using Nystrom, FastGaussQuadrature, Plots, LinearAlgebra, LaTeXStrings
using Nystrom: error_interior_green_identity, error_exterior_green_identity

function fig_gen()
    dim = 3
    Γ = Domain(dim=dim)
    geo = sphere()
    push!(Γ,geo)

    fig       = plot(yscale=:log10,xscale=:log10,xlabel= "√dof",ylabel="error")
    qorder    = (2,3,4)
    h0        = 2.0
    niter     = 4
    # operators = (Laplace(dim=dim),Helmholtz(dim=dim,k=2π))
    operators = (Helmholtz(dim=dim,k=1),)
    for op in operators
        # construct interior solution
        xout = (3,3,3)
        u    = (x)   -> SingleLayerKernel(op)(xout,x)
        dudn = (x,n) -> DoubleLayerKernel(op)(xout,x,n)
        for p in qorder
            conv_order = p+1
            meshgen!(Γ,h0)
            ee_interior = []
            dof         = []
            for _ in 1:niter
                quadgen!(Γ,p,algo1d=gausslegendre)
                S  = SingleLayerOperator(op,Γ)
                D  = DoubleLayerOperator(op,Γ)
                δS = GreensCorrection(S)
                δD = GreensCorrection(D)
                # test interior Green identity
                γ₀u       = γ₀(u,Γ)
                γ₁u       = γ₁(dudn,Γ)
                ee = error_interior_green_identity(S+δS,D+δD,γ₀u,γ₁u)
                push!(ee_interior,norm(ee,Inf)/norm(γ₀u,Inf))
                @show length(Γ)
                push!(dof,length(Γ))
                @show sum(Γ.quadrature.weights)
                refine!(Γ)
            end
            dof = sqrt.(dof) # roughly the inverse of the mesh size
            plot!(fig,dof,ee_interior,label=Nystrom.getname(op)*": p=$p",m=:o)
            plot!(fig,dof,1 ./(dof.^conv_order)*dof[end]^(conv_order)*ee_interior[end],
                  label="",linewidth=4)
        end
    end
    return fig
end

fig = fig_gen()
fname = "/home/lfaria/Dropbox/Luiz-Carlos/general_regularization/draft/figures/fig2a.pdf"
savefig(fig,fname)
display(fig)
