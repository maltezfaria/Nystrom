using Nystrom, FastGaussQuadrature, Plots, LinearAlgebra, IterativeSolvers, ParametricSurfaces, LinearMaps, LaTeXStrings
using Nystrom: Point

function convergence_exterior_dirichlet3d()
    pde = Helmholtz(dim=3,k=π)
    geo = Acorn()
    fig = plot(yscale=:log10,xscale=:log10,xlabel= L"\sqrt{N}",ylabel="error",legend=:bottomleft,
               framestyle=:box,xtickfontsize=10,ytickfontsize=10)
    # construct exterior solution
    qorder  = (2,3,4)
    h0      = 0.25
    el_size = h0 .* qorder
    niter   = (5,5,5) .+ 3
    xin     = Point(0,0,0)
    c    = 1
    u    = (x)   -> SingleLayerKernel(pde)(xin,x)*c
    dudn = (x,n) -> transpose(DoubleLayerKernel(pde)(xin,x,n))*c
    # far field
    Xfar = [5*Point(sin(θ)*cos(ϕ),sin(θ)*sin(ϕ),cos(θ)) for θ in 0:0.1:π, ϕ in 0:0.1:2π] |> vec
    Ufar = [u(x) for x in Xfar]
    cc = 0
    for (p,h) in zip(qorder,el_size)
        cc+=1
        conv_dtn_order = p - 1
        conv_far_order = p + 1
        eDtN  = []
        eFar  = []
        dof = []
        for n in 1:1:niter[cc]
            meshgen!(geo,h/n)
            Γ = quadgen(geo,p,gausslegendre)
            S,D = single_double_layer(pde,Γ)
            # solve exterior problem
            γ₀u       = γ₀(u,Γ)
            γ₁u       = γ₁(dudn,Γ)
            η         = im*pde.k
            L         = LinearMap(σ -> σ/2 + D*σ - η*(S*σ),length(γ₀u))
            σ,ch      = gmres(L,γ₀u,verbose=false,log=true,tol=1e-14,restart=1000,maxiter=1000)
            # compute the error in the other trace (i.e. the DtN map)
            S,D = nothing, nothing
            Base.GC.gc()
            # ADL,H = adjointdoublelayer_hypersingular(pde,Γ)
            # push!(eDtN,norm(γ₁u - (H*σ - η*(ADL*σ - σ/2)),Inf))
            # compute the error on the "far field""
            S,D = single_double_layer(pde,Xfar,Γ,correction=:nothing)
            push!(eFar,norm(D*σ - η*(S*σ) - Ufar,Inf)/norm(Ufar,Inf))
            @show length(Γ), eFar[end], ch.iters
            push!(dof,length(Γ))
            end
        dof_per_dim = sqrt.(dof)
        # convergence figure
        # plot!(fig,dof_per_dim,eDtN,label="DtN error for p=$p",m=:o,color=cc)
        # plot!(fig,dof_per_dim,1 ./(dof_per_dim.^conv_dtn_order)*dof_per_dim[end]^(conv_dtn_order)*eDtN[end],
        #       label="",linewidth=4,line=:dot,color=cc)
        plot!(fig,dof_per_dim,eFar,label="Helmholtz: p=$p",m=:x,color=cc)
        plot!(fig,dof_per_dim,1 ./(dof_per_dim.^conv_far_order)*dof_per_dim[end]^(conv_far_order)*eFar[end],
              label="",linewidth=4,line=:dot,color=cc)
    end
    return fig
end

# figure 3b
fig       = convergence_exterior_dirichlet3d()
fname     = "/home/lfaria/Dropbox/Luiz-Carlos/general_regularization/draft/figures/acorn_dirichlet_convergence.pdf"
savefig(fig,fname)
display(fig)
