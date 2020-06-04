using Nystrom, FastGaussQuadrature, Plots, LinearAlgebra, IterativeSolvers, ParametricSurfaces, LaTeXStrings, LinearMaps
using Nystrom: sphere_helmholtz_soundsoft, error_exterior_green_identity, Point

function fig_gen()
    dim = 3
    R = 1
    geo = Sphere(radius=R)
    fig       = plot(yscale=:log10,xscale=:log10,xlabel=L"\sqrt{N}",ylabel="error",
                     framestyle=:box,xtickfontsize=10,ytickfontsize=10)
    qorder    = (2,3,4)
    h0        = 0.5
    el_size   = h0 .* (qorder)
    niter     = (5,4,5)
    k         = π
    pde       = Helmholtz(dim=dim,k=k)
    θ         = π/2
    ϕ         = 0
    ue(x) = sphere_helmholtz_soundsoft(x;radius=R,k=k,θin=θ,ϕin = ϕ)
    xtest = [Point(2*R*sin(θ)*cos(ϕ),2*R*sin(θ)*sin(ϕ),2*R*cos(θ)) for θ in 0:0.1:π, ϕ in 0:0.1:2π]
    Ue    = [ue(x) for x in xtest]
    kx = k*sin(θ)*cos(ϕ)
    ky = k*sin(θ)*sin(ϕ)
    kz = k*cos(θ)
    cc = 0
    for (p,h) in zip(qorder,el_size)
        cc+=1
        conv_order = p + 1
        meshgen!(geo,h)
        gmres_iter = []
        dof        = []
        ee         = []
        for _ in 1:niter[cc]
            Γ = quadgen(geo,p,gausslegendre)
            S,D = single_double_layer(pde,Γ)
            γ₀U   = γ₀( (y) -> -exp(im*(kx*y[1] + ky*y[2] +  kz*y[3])), Γ)
            L     = LinearMap(σ -> σ/2 + D*σ - im*k*(S*σ),length(γ₀U))
            σ,ch  = gmres(L,γ₀U,verbose=false,log=true,tol=1e-14,restart=100)
            ua(x) = DoubleLayerPotential(pde,Γ)(σ,x) - im*k*SingleLayerPotential(pde,Γ)(σ,x)
            Ua    = [ua(x) for x in xtest]
            push!(ee,norm(Ue.-Ua,Inf)/norm(Ue,Inf))
            push!(dof,length(Γ))
            @show length(Γ), ee[end], ch.iters
            refine!(geo)
        end
        dof_per_dim = sqrt.(dof) # dof per dimension, roughly inverse of mesh size
        plot!(fig,dof_per_dim,ee,label=Nystrom.getname(pde)*": p=$p",m=:o,color=p)
        # plot!(fig,dof_per_dim,1 ./(dof_per_dim.^conv_order)*dof_per_dim[end]^(conv_order)*ee[end],
        #       label="",linewidth=4,color=p)
    end
    return fig
end

fig = fig_gen()
fname = "/home/lfaria/Dropbox/Luiz-Carlos/general_regularization/draft/figures/fig4b.pdf"
savefig(fig,fname)
display(fig)
