using Nystrom, FastGaussQuadrature, Plots, LinearAlgebra, LaTeXStrings
using Nystrom: error_interior_green_identity, error_exterior_green_identity

function fig_gen()
    dim = 2
    Γ = Domain(dim=dim)
    geo = circle()
    push!(Γ,geo)

    fig       = plot(yscale=:log10,xscale=:log10,xlabel="#dof",ylabel="error")
    qorder    = (2,3,4)
    h0        = 0.1
    niter     = 4
    operators = (Elastostatic(dim=2,μ=1,λ=1),Elastodynamic(dim=2,λ=1,μ=1,ρ=1,ω=1))
    op = operators[1]
    p  = qorder[1]
    for op in operators
        # construct interior solution
        xout = (4,3)
        u    = (x)   -> SingleLayerKernel(op)(xout,x)
        dudn = (x,n) -> DoubleLayerKernel(op)(xout,x,n)
        # construct exterior solution
        xin  = (-.1,0.2)
        v    = (x)   -> SingleLayerKernel(op)(xin,x)
        dvdn = (x,n) -> DoubleLayerKernel(op)(xin,x,n)
        for p in qorder
            meshgen!(Γ,h0)
            ee_interior = []
            ee_exterior = []
            dof         = []
            for _ in 1:niter
                quadgen!(Γ,(p,),gausslegendre)
                S  = SingleLayerOperator(op,Γ)
                D  = DoubleLayerOperator(op,Γ)
                δS = GreensCorrection(S)
                δD = GreensCorrection(D)
                # test interior Green identity
                γ₀u       = γ₀(u,Γ)
                γ₁u       = γ₁(dudn,Γ)
                ee = error_interior_green_identity(S+δS,D+δD,γ₀u,γ₁u)
                push!(ee_interior,norm(ee,Inf)/norm(γ₀u,Inf))
                # test exterior Green identity
                γ₀v       = γ₀(v,Γ)
                γ₁v       = γ₁(dvdn,Γ)
                ee = error_exterior_green_identity(S+δS,D+δD,γ₀v,γ₁v)
                push!(ee_exterior,norm(ee,Inf)/norm(γ₀v,Inf))
                push!(dof,length(Γ))
                refine!(Γ)
            end
            plot!(fig,dof,ee_interior,label=Nystrom.getname(op)*": p=$p",m=:x)
            # plot!(fig,dof,ee_exterior,label="op = $op, p=$p",m=:o)
        end
    end
    return fig
end

fig = fig_gen()
fname = "/home/lfaria/Dropbox/Luiz-Carlos/general_regularization/draft/figures/fig1a.svg"
savefig(fig,fname)
display(fig)
