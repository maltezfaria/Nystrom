using Nystrom, FastGaussQuadrature, Plots, LinearAlgebra
using Nystrom: error_interior_green_identity, error_interior_derivative_green_identity

# figure2
dim       = 3
qorder    = (2,3,4)
h         = 2.0
niter     = 5

#2a
operator  = Helmholtz(dim=dim,k =1)
fig       = convergence_interior_greens_identity(operator,dim,qorder,h,niter;derivative=false)
fname     = "/home/lfaria/Dropbox/Luiz-Carlos/general_regularization/draft/figures/fig2a.pdf"
savefig(fig,fname)

#2b
operator = Helmholtz(dim =dim,k =1)
fig       = convergence_interior_greens_identity(operator,dim,qorder,h,niter;derivative=true)
fname = "/home/lfaria/Dropbox/Luiz-Carlos/general_regularization/draft/figures/fig2b.pdf"
savefig(fig,fname)

#2c
operator = Elastodynamic(dim =dim,μ =1,ρ =1,λ =1,ω =1)
fig       = convergence_interior_greens_identity(operator,dim,qorder,h,niter;derivative=false)
fname = "/home/lfaria/Dropbox/Luiz-Carlos/general_regularization/draft/figures/fig2c.pdf"
savefig(fig,fname)

#2d
operator = Elastodynamic(dim=dim,μ =1,ρ =1,λ =1,ω =1)
fig       = convergence_interior_greens_identity(operator,dim,qorder,h,niter;derivative=true)
fname = "/home/lfaria/Dropbox/Luiz-Carlos/general_regularization/draft/figures/fig2d.pdf"
savefig(fig,fname)
