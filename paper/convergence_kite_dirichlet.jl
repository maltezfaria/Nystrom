using Nystrom, FastGaussQuadrature, Plots, LinearAlgebra, IterativeSolvers, ParametricSurfaces, LinearMaps, LaTeXStrings

function convergence_exterior_dirichlet(pde,dim,qorder,h0,niter)
    geo = Kite()
    fig = plot(yscale=:log10,xscale=:log10,xlabel= dim==2 ? L"N" : L"\sqrt{N}",ylabel="error",legend=:bottomleft,
               framestyle=:box,xtickfontsize=10,ytickfontsize=10)
    # construct exterior solution
    xin  = 0.0*ones(dim)
    c    = 1
    u    = (x)   -> SingleLayerKernel(pde)(xin,x)*c
    dudn = (x,n) -> transpose(DoubleLayerKernel(pde)(xin,x,n))*c
    # far field
    Xfar = [5 .* (cos(th),sin(th)) for th in 0:0.1:2π]
    Ufar = [u(x) for x in Xfar]
    cc = 0
    for p in qorder
        cc+=1
        conv_dtn_order = p - 1
        conv_far_order = p + 1
        meshgen!(geo,p*h0)
        eDtN  = []
        eFar  = []
        dof = []
        for n in 1:niter
            Γ = quadgen(geo,p,gausslegendre)
            S,D = single_double_layer(pde,Γ)
            # solve exterior problem
            γ₀u       = γ₀(u,Γ)
            γ₁u       = γ₁(dudn,Γ)
            η         = im*pde.k
            L         = LinearMap(σ -> σ/2 + D*σ - η*(S*σ),length(γ₀u))
            σ,ch      = gmres(L,γ₀u,verbose=false,log=true,tol=1e-14,restart=100)
            # # compute the error in the other trace (i.e. the DtN map)
            # ADL,H = adjointdoublelayer_hypersingular(pde,Γ)
            # push!(eDtN,norm(γ₁u - (H*σ - η*(ADL*σ - σ/2)),Inf)/norm(γ₁u,Inf))
            # compute the error on the "far field""
            S,D = single_double_layer(pde,Xfar,Γ,correction=:nothing)
            push!(eFar,(norm(D*σ - η*(S*σ) - Ufar,Inf))/norm(Ufar,Inf))
            @show length(Γ), eFar[end], ch.iters
            push!(dof,length(Γ))
            meshgen!(geo,p*h0/n)
        end
        dof_per_dim = dof
        # convergence figure
        # plot!(fig,dof_per_dim,eDtN,label="DtN error for p=$p",m=:o,color=cc)
        # plot!(fig,dof_per_dim,1 ./(dof_per_dim.^conv_dtn_order)*dof_per_dim[end]^(conv_dtn_order)*eDtN[end],
        #       label="",linewidth=4,line=:dot,color=cc)
        plot!(fig,dof_per_dim,abs.(eFar),label="Helmholtz: p=$p",m=:x,color=cc)
        plot!(fig,dof_per_dim,1 ./(dof_per_dim.^conv_far_order)*dof_per_dim[end]^(conv_far_order)*eFar[end],
              label="",linewidth=4,line=:dot,color=cc)
    end
    return fig
end

# figure 3a
dim       = 2
qorder    = (2,3,4,5,6)
h         = 0.05
niter     = 13
pde       = Helmholtz(dim=dim,k=π)
fig       = convergence_exterior_dirichlet(pde,dim,qorder,h,niter)
fname     = "/home/lfaria/Dropbox/Luiz-Carlos/general_regularization/draft/figures/kite_dirichlet_convergence.pdf"
savefig(fig,fname)
display(fig)
