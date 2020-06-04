using Nystrom, FastGaussQuadrature, Plots, LinearAlgebra, IterativeSolvers, ParametricSurfaces, LinearMaps, LaTeXStrings
using Nystrom: Point
using StaticArrays

function convergence_exterior_neumann3d()
    pde = Elastodynamic(dim=3,μ=1,λ=2,ρ=1,ω=2π)
    fig1 = plot(yscale=:log10,xscale=:log10,xlabel= L"\sqrt{N}",ylabel="error",legend=:topright,
               framestyle=:box,xtickfontsize=10,ytickfontsize=10)
    fig2 = plot(xlabel= L"N",ylabel="Number of iterations",legend=:topright,
                framestyle=:box,xtickfontsize=10,ytickfontsize=10)
    # construct exterior solution
    p       = 3
    h       = 1
    niter   = 6
    xin     = Point(0,-0.25,0)
    c       = ones(3)
    u    = (x)   -> SingleLayerKernel(pde)(xin,x)*c
    dudn = (x,n) -> transpose(DoubleLayerKernel(pde)(xin,x,n))*c
    # far field
    Xfar = [5*Point(sin(θ)*cos(ϕ),sin(θ)*sin(ϕ),cos(θ)) for θ in 0:0.1:π, ϕ in 0:0.1:2π] |> vec
    Ufar = [u(x) for x in Xfar]
    Umax = norm(norm.(Ufar,Inf),Inf)
    cc = 0
    shapes = (Sphere(),Ellipsoid(paxis=(2,1,1)),Bean())
    # shapes = (Sphere(),)
    names  = ("sphere","ellipsoid","bean")
    dofs = []
    ee   = []
    for geo in shapes
        cc+=1
        eFar  = []
        dof = []
        niters = []
        for n in 1:niter
            meshgen!(geo,h/n)
            Γ = quadgen(geo,p,gausslegendre)
            ADL,H = adjointdoublelayer_hypersingular(pde,Γ)
            # solve exterior problem
            γ₀u       = γ₀(u,Γ)
            γ₁u       = γ₁(dudn,Γ)
            η         = im*pde.ω
            L         = LinearMap(3*length(γ₀u)) do x
                T = eltype(x)
                σ = reinterpret(SVector{3,T},x)
                reinterpret(T,η*σ/2 - η*(ADL*σ) + H*σ )
            end
            σ,ch      = gmres(L,reinterpret(ComplexF64,γ₁u),verbose=false,log=true,tol=1e-8,restart=1000,maxiter=1000)
            σ         = reinterpret(SVector{3,ComplexF64},σ)
            ADL,H = nothing, nothing
            Base.GC.gc()
            # compute the error on the "far field""
            S,D = single_double_layer(pde,Xfar,Γ,correction=:nothing)
            push!(eFar,(norm(D*σ - η*(S*σ) - Ufar,Inf))/Umax)
            push!(niters,ch.iters)
            @show length(Γ), eFar[end], ch.iters
            push!(dof,length(Γ))
        end
        dof_per_dim = sqrt.(dof)
        # convergence figure
        plot!(fig1,dof_per_dim,eFar,label=names[cc], m=:x,color=cc)
        plot!(fig2,dof_per_dim,niters,label=names[cc], m=:o,color=cc)
        if geo == shapes[end]
            conv_order = 3
            plot!(fig1,dof_per_dim,1 ./(dof_per_dim.^conv_order)*dof_per_dim[end]^(conv_order)*eFar[end],
                  label="",linewidth=4,line=:dot,color=cc)
        end
    end
    return fig1,fig2
end

fig1,fig2       = convergence_exterior_neumann3d()
fname     = "/home/lfaria/Dropbox/Luiz-Carlos/general_regularization/draft/figures/elastody_neum_er.pdf"
savefig(fig1,fname)
fname     = "/home/lfaria/Dropbox/Luiz-Carlos/general_regularization/draft/figures/elastody_neum_iters.pdf"
savefig(fig2,fname)
