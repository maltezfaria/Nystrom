using Nystrom, FastGaussQuadrature, Plots, LinearAlgebra, IterativeSolvers, ParametricSurfaces
using Nystrom: error_interior_green_identity, error_exterior_green_identity

function convergence_exterior_dirichlet(pde,dim,qorder,h0,niter)
    geo = dim == 2 ? Kite() : Sphere()
    fig = plot(yscale=:log10,xscale=:log10,xlabel= dim==2 ? "N" : "√N",ylabel="error",legend=:bottomleft)
    # construct exterior solution
    xin  = 0.3*ones(dim)
    c    = pde isa Union{Helmholtz,Laplace} ? 1 : ones(dim)
    u    = (x)   -> SingleLayerKernel(pde)(xin,x)*c
    dudn = (x,n) -> transpose(DoubleLayerKernel(pde)(xin,x,n))*c
    # far field
    Xfar = dim == 2 ? [5 .* (cos(th),sin(th)) for th in 0:0.1:2π] : [5 .* (cos(th),sin(th),0.0) for th in 0:0.1:2π]
    Ufar = [u(x) for x in Xfar]
    for p in qorder
        conv_dtn_order = p - 1
        conv_far_order = p + 1
        meshgen!(geo,h0)
        eDtN  = []
        eFar  = []
        dof = []
        for _ in 1:niter
            Γ = quadgen(geo,p,gausslegendre)
            S,D = single_double_layer(pde,Γ)
            # solve exterior problem
            γ₀u       = γ₀(u,Γ)
            γ₁u       = γ₁(dudn,Γ)
            rhs       = S*γ₁u
            η     = im*pde.k
            σ,ch  = gmres(I/2 + D - η*S,γ₀u,verbose=false,log=true,tol=1e-14,restart=100)
            # compute the error in the other trace (i.e. the DtN map)
            ADL,H = adjointdoublelayer_hypersingular(pde,Γ)
            push!(eDtN,norm(γ₁u - (H*σ - η*(ADL*σ - σ/2)),Inf))
            # compute the error on the "far field""
            S,D = single_double_layer(pde,Xfar,Γ,correction=:nothing)
            push!(eFar,norm((D - η*S)*σ - Ufar,Inf))
            @show length(Γ),eDtN[end], eFar[end], ch.iters
            push!(dof,length(Γ))
            refine!(geo)
        end
        dof_per_dim = dim == 2 ? dof : sqrt.(dof)
        # convergence figure
        plot!(fig,dof_per_dim,eDtN,label="DtN error for p=$p",m=:o,color=conv_dtn_order)
        plot!(fig,dof_per_dim,1 ./(dof_per_dim.^conv_dtn_order)*dof_per_dim[end]^(conv_dtn_order)*eDtN[end],
              label="",linewidth=4,line=:dot,color=conv_dtn_order)
        plot!(fig,dof_per_dim,eFar,label="Farfield error for p=$p",m=:x,color=conv_far_order)
        plot!(fig,dof_per_dim,1 ./(dof_per_dim.^conv_far_order)*dof_per_dim[end]^(conv_far_order)*eFar[end],
              label="",linewidth=4,line=:dot,color=conv_dtn_order)
    end
    return fig
end

# figure 3a
dim       = 2
qorder    = (2,3,4)
h         = 0.1
niter     = 6
pde        = Helmholtz(dim=dim,k=2π)
fig       = convergence_exterior_dirichlet(pde,dim,qorder,h,niter)
fname     = "/home/lfaria/Dropbox/Luiz-Carlos/general_regularization/draft/figures/fig3a.pdf"
savefig(fig,fname)
display(fig)

# figure 3b
dim       = 3
qorder    = (2,3,4)
h         = 1
niter     = 3
pde        = Helmholtz(dim=dim,k=π)
fig       = convergence_exterior_dirichlet(pde,dim,qorder,h,niter)
fname     = "/home/lfaria/Dropbox/Luiz-Carlos/general_regularization/draft/figures/fig3b.pdf"
savefig(fig,fname)
display(fig)
