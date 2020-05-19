using Nystrom, FastGaussQuadrature, Plots, LinearAlgebra
using Nystrom: error_interior_green_identity, error_exterior_green_identity

function fig_gen()
    dim = 2
    Γ1 = Domain(dim=dim)
    Γ2 = Domain(dim=dim)
    c1 = circle()
    c2 = circle(center=(2+1e-2,0))
    push!(Γ1,c1)
    push!(Γ2,c2)
    fig       = plot(yscale=:log10,xscale=:log10,xlabel="N",ylabel="error")
    qorder    = (2,3,4)
    h0        = 1
    niter     = 6
    operators = (Helmholtz(dim=2,k=1),)
    # operators = (Helmholtz(dim=dim,k=1),)
    for op in operators
        # construct interior solution
        xout = (3,3)
        u    = (x)   -> SingleLayerKernel(op)(xout,x)
        dudn = (x,n) -> DoubleLayerKernel(op)(xout,x,n)
        for p in qorder
            conv_order = p + 1
            meshgen!(Γ1,h0)
            meshgen!(Γ2,h0)
            ee_interior = []
            ee_exterior = []
            dof         = []
            for _ in 1:niter
                quadgen!(Γ1,p;algo1d=gausslegendre)
                quadgen!(Γ2,p;algo1d=gausslegendre)
                # blocks
                S11  = SingleLayerOperator(op,Γ1,Γ1)
                S12  = SingleLayerOperator(op,Γ1,Γ2)
                S21  = SingleLayerOperator(op,Γ2,Γ1)
                S22  = SingleLayerOperator(op,Γ2,Γ2)
                D11  = DoubleLayerOperator(op,Γ1,Γ1)
                D12  = DoubleLayerOperator(op,Γ1,Γ2)
                D21  = DoubleLayerOperator(op,Γ2,Γ1)
                D22  = DoubleLayerOperator(op,Γ2,Γ2)
                # corrections
                δS11 = GreensCorrection(S11)
                δS12 = GreensCorrection(S12)
                δS21 = GreensCorrection(S21)

                δS22 = GreensCorrection(S22)
                δD11 = GreensCorrection(D11)
                δD12 = GreensCorrection(D12)
                δD21 = GreensCorrection(D21)
                δD22 = GreensCorrection(D22)
                # construct global operator
                S    = [S11+δS11 S12+δS12;
                        S21+δS21 S22+δS22]
                D    = [D11+δD11 D12+δD12;
                        D21+δD21 D22+δD22]
                # S = SingleLayerOperator(op,Γ)
                # S += GreensCorrection(S)
                # D = DoubleLayerOperator(op,Γ)
                # D += GreensCorrection(D)
                # test interior Green identity
                γ₀u1   = γ₀(u,Γ1)
                γ₀u2   = γ₀(u,Γ2)
                γ₁u1   = γ₁(dudn,Γ1)
                γ₁u2   = γ₁(dudn,Γ2)
                γ₀u    = [γ₀u1;γ₀u2]
                γ₁u    = [γ₁u1;γ₁u2]
                ee         = error_interior_green_identity(S,D,γ₀u,γ₁u)
                # ee         = error_interior_green_identity(S11+δS11,D11+δD11,γ₀u1,γ₁u1)
                # ee         = error_interior_green_identity(S22+δS22,D22+δD22,γ₀u1,γ₁u1)
                push!(ee_interior,norm(ee,Inf)/norm(γ₀u,Inf))
                push!(dof,length(Γ1)+length(Γ2))
                @show dof[end],ee_interior[end]
                refine!(Γ1)
                refine!(Γ2)
            end
            plot!(fig,dof,ee_interior,label=Nystrom.getname(op)*": p=$p",m=:o)
            plot!(fig,dof,1 ./(dof.^conv_order)*dof[end]^(conv_order)*ee_interior[end],
                  label="",linewidth=4)
        end
    end
    return fig
end

fig = fig_gen()
fname = "/home/lfaria/Dropbox/Luiz-Carlos/general_regularization/draft/figures/fig7.pdf"
savefig(fig,fname)
display(fig)
