using Nystrom, FastGaussQuadrature, Plots, LinearAlgebra, ParametricSurfaces
using Nystrom: error_interior_green_identity, error_interior_derivative_green_identity

function convergence_interior_greens_identity(op,dim,qorder,h,niter;derivative=false)
    # Γ = Domain(dim=dim)
    geo = dim == 2 ? Circle() : Sphere()
    # push!(Γ,geo)
    # figs      = []
    fig       = plot(yscale=:log10,xscale=:log10,xlabel="N",ylabel="error",legend=:bottomleft)
    cc = 1
    colors = [:red,:green,:blue,:yellow,:black,:pink]
    # construct interior solution
    xout = 3 * ones(dim)
    c    = op isa Helmholtz ? 1 : ones(dim)
    u    = (x)   -> SingleLayerKernel(op)(xout,x)*c
    dudn = (x,n) -> transpose(DoubleLayerKernel(op)(xout,x,n))*c
    for p in qorder
        if op isa Helmholtz
            conv_order = derivative ? p-1 : p+1
        else
            conv_order = derivative ? p-1 : p
        end
        meshgen!(geo,h)
        dof         = []
        ee_interior = []
        for _ in 1:niter
            Γ = quadgen(geo,p,gausslegendre)
            γ₀u       = γ₀(u,Γ)
            γ₁u       = γ₁(dudn,Γ)
            if derivative==false
                S,D   = single_double_layer(op,Γ)
                ee    = error_interior_green_identity(S,D,γ₀u,γ₁u)
            else
                ADL,H = adjointdoublelayer_hypersingular(op,Γ)
                ee    = error_interior_derivative_green_identity(ADL,H,γ₀u,γ₁u)
            end
            # test interior Green identity
            push!(ee_interior,norm(ee,Inf)/norm(γ₀u,Inf))
            push!(dof,length(Γ))
            refine!(geo)
        end
        dof_per_dim = dim == 2 ? dof : sqrt.(dof)
        plot!(fig,dof_per_dim,ee_interior,label=Nystrom.getname(op)*": p=$p",m=:o,color=colors[cc])
        plot!(fig,dof_per_dim,1 ./(dof_per_dim.^conv_order)*dof_per_dim[end]^(conv_order)*ee_interior[end],
              label="",linewidth=4,line=:dot,color=colors[cc])
        cc += 1
    end
    return fig
end


# figure1
dim       = 2
qorder    = (2,3,4)
h         = 1.0
niter     = 6

#1a
operators = Helmholtz(dim =dim,k =1)
fig       = convergence_interior_greens_identity(operators,dim,qorder,h,niter;derivative=false)
fname = "/home/lfaria/Dropbox/Luiz-Carlos/general_regularization/draft/figures/fig1a.pdf"
savefig(fig,fname)

# 1b
operators = Helmholtz(dim =dim,k =1)
fig       = convergence_interior_greens_identity(operators,dim,qorder,h,niter;derivative=true)
fname = "/home/lfaria/Dropbox/Luiz-Carlos/general_regularization/draft/figures/fig1b.pdf"
savefig(fig,fname)

# 1c
operators = Elastodynamic(dim =dim,μ =1,ρ =1,λ =1,ω =1)
fig       = convergence_interior_greens_identity(operators,dim,qorder,h,niter;derivative=false)
fname = "/home/lfaria/Dropbox/Luiz-Carlos/general_regularization/draft/figures/fig1c.pdf"
savefig(fig,fname)

# figure1d
operators = Elastodynamic(dim =dim,μ =1,ρ =1,λ =1,ω =1)
fig       = convergence_interior_greens_identity(operators,dim,qorder,h,niter;derivative=true)
fname = "/home/lfaria/Dropbox/Luiz-Carlos/general_regularization/draft/figures/fig1d.pdf"
savefig(fig,fname)