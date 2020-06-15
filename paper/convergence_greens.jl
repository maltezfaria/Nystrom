using Nystrom, FastGaussQuadrature, Plots, LinearAlgebra, ParametricSurfaces, LaTeXStrings
using Nystrom: error_interior_green_identity, error_interior_derivative_green_identity

function convergence_interior_greens_identity(op,dim,qorder,h,niter;derivative=false)
    geo = dim == 2 ? Kite() : Bean()
    fig       = plot(yscale=:log10,xscale=:log10,
                     xlabel= dim==2 ? L"N" : L"\sqrt{N}",ylabel="error",legend=:bottomleft,
                     framestyle=:box,xtickfontsize=10,ytickfontsize=10)
    cc = 1
    colors = [:red,:green,:blue,:yellow,:black,:pink]
    # construct interior solution
    xout = 3 * ones(dim)
    c    = ones(Nystrom.default_density_eltype(op))
    u    = (x)   -> SingleLayerKernel(op)(xout,x)*c
    dudn = (x,n) -> transpose(DoubleLayerKernel(op)(xout,x,n))*c
    for p in qorder
        if op isa Union{Helmholtz,Laplace}
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
        if dim == 2
            plot!(fig,dof_per_dim,
                  log10.(dof_per_dim)./(dof_per_dim.^conv_order)*dof_per_dim[end]^(conv_order)/log10(dof_per_dim[end])*ee_interior[end],
                  label="",linewidth=4,line=:dot,color=colors[cc])
        else
            plot!(fig,dof_per_dim,1 ./(dof_per_dim.^conv_order)*dof_per_dim[end]^(conv_order)*ee_interior[end],
                  label="",linewidth=4,line=:dot,color=colors[cc])
        end
        cc += 1
    end
    return fig
end

# 2D cases
dim       = 2
qorder    = (2,3,4)
h         = 1.0
niter     = 8

#2D scalar
operators = Laplace(dim =dim)
fig       = convergence_interior_greens_identity(operators,dim,qorder,h,niter;derivative=false)
fname = "/home/lfaria/Dropbox/Luiz-Carlos/general_regularization/draft/figures/greens_laplace_2d.pdf"
geo = Kite()
meshgen!(geo,0.02)
plot!(geo,lc=:black,inset =  (1, bbox(0.7, 0.05, 0.25, 0.25, :top)), subplot = 2,aspect_ratio=:equal,framestyle=:box)
savefig(fig,fname)

operators = Helmholtz(dim=dim,k=1)
fig       = convergence_interior_greens_identity(operators,dim,qorder,h,niter;derivative=false)
fname = "/home/lfaria/Dropbox/Luiz-Carlos/general_regularization/draft/figures/greens_helmholtz_2d.pdf"
savefig(fig,fname)

operators = Laplace(dim =dim)
fig       = convergence_interior_greens_identity(operators,dim,qorder,h,niter;derivative=true)
fname = "/home/lfaria/Dropbox/Luiz-Carlos/general_regularization/draft/figures/greensp_laplace_2d.pdf"
savefig(fig,fname)

operators = Helmholtz(dim =dim,k =1)
fig       = convergence_interior_greens_identity(operators,dim,qorder,h,niter;derivative=true)
fname = "/home/lfaria/Dropbox/Luiz-Carlos/general_regularization/draft/figures/greensp_helmholtz_2d.pdf"
savefig(fig,fname)

# 2D vectorial problems
operators = Elastostatic(dim=dim,μ=1,λ=2)
fig       = convergence_interior_greens_identity(operators,dim,qorder,h,niter;derivative=false)
fname = "/home/lfaria/Dropbox/Luiz-Carlos/general_regularization/draft/figures/greens_elastostatic_2d.pdf"
savefig(fig,fname)

operators = Elastodynamic(dim=dim,μ=1,ρ=1,λ=2,ω=1)
fig       = convergence_interior_greens_identity(operators,dim,qorder,h,niter;derivative=false)
fname = "/home/lfaria/Dropbox/Luiz-Carlos/general_regularization/draft/figures/greens_elastodynamic_2d.pdf"
savefig(fig,fname)

operators = Elastostatic(dim=dim,μ=1,λ=2)
fig       = convergence_interior_greens_identity(operators,dim,qorder,h,niter;derivative=true)
fname = "/home/lfaria/Dropbox/Luiz-Carlos/general_regularization/draft/figures/greensp_elastostatic_2d.pdf"
savefig(fig,fname)

operators = Elastodynamic(dim=dim,μ=1,ρ=1,λ=2,ω=1)
fig       = convergence_interior_greens_identity(operators,dim,qorder,h,niter;derivative=true)
fname = "/home/lfaria/Dropbox/Luiz-Carlos/general_regularization/draft/figures/greensp_elastodynamic_2d.pdf"
savefig(fig,fname)

# 3D
dim       = 3
qorder    = (2,3,4)
h         = 2.0
niter     = 4

# Scalar
operator  = Laplace(dim=dim)
fig       = convergence_interior_greens_identity(operator,dim,qorder,h,niter;derivative=false)
# geo       = Bean()
# meshgen!(geo,0.1)
# plot!(geo,lc=:black,inset =  (1, bbox(0.6, 0.05, 0.35, 0.35, :top)),
#       subplot = 2,aspect_ratio=:equal,framestyle=:box,axis=nothing)
fname     = "/home/lfaria/Dropbox/Luiz-Carlos/general_regularization/draft/figures/greens_laplace_3d.pdf"
savefig(fig,fname)

operator  = Helmholtz(dim=dim,k =1)
fig       = convergence_interior_greens_identity(operator,dim,qorder,h,niter;derivative=false)
fname     = "/home/lfaria/Dropbox/Luiz-Carlos/general_regularization/draft/figures/greens_helmholtz_3d.pdf"
savefig(fig,fname)

operator  = Laplace(dim=dim)
fig       = convergence_interior_greens_identity(operator,dim,qorder,h,niter;derivative=true)
fname     = "/home/lfaria/Dropbox/Luiz-Carlos/general_regularization/draft/figures/greensp_laplace_3d.pdf"
savefig(fig,fname)

operator  = Helmholtz(dim=dim,k =1)
fig       = convergence_interior_greens_identity(operator,dim,qorder,h,niter;derivative=true)
fname     = "/home/lfaria/Dropbox/Luiz-Carlos/general_regularization/draft/figures/greensp_helmholz_3d.pdf"
savefig(fig,fname)

# Vectorial
operator = Elastostatic(dim =dim,μ =1,λ =2)
fig       = convergence_interior_greens_identity(operator,dim,qorder,h,niter;derivative=false)
fname = "/home/lfaria/Dropbox/Luiz-Carlos/general_regularization/draft/figures/greens_elastostatic_3d.pdf"
savefig(fig,fname)

operator = Elastodynamic(dim =dim,μ =1,ρ =1,λ =2,ω =1)
fig       = convergence_interior_greens_identity(operator,dim,qorder,h,niter;derivative=false)
fname = "/home/lfaria/Dropbox/Luiz-Carlos/general_regularization/draft/figures/greens_elastodynamic_3d.pdf"
savefig(fig,fname)

operator = Elastostatic(dim =dim,μ =1,λ =2)
fig       = convergence_interior_greens_identity(operator,dim,qorder,h,niter;derivative=true)
fname = "/home/lfaria/Dropbox/Luiz-Carlos/general_regularization/draft/figures/greensp_elastostatic_3d.pdf"
savefig(fig,fname)

operator = Elastodynamic(dim =dim,μ =1,ρ =1,λ =2,ω =1)
fig       = convergence_interior_greens_identity(operator,dim,qorder,h,niter;derivative=true)
fname = "/home/lfaria/Dropbox/Luiz-Carlos/general_regularization/draft/figures/greensp_elastodynamic_3d.pdf"
savefig(fig,fname)

# # 2d
# operators = Elastodynamic(dim =dim,μ =1,ρ =1,λ =2,ω =1)
# fig       = convergence_interior_greens_identity(operators,dim,qorder,h,niter;derivative=true)
# fname = "/home/lfaria/Dropbox/Luiz-Carlos/general_regularization/draft/figures/greensp_elastody_3d.pdf"
# savefig(fig,fname)
