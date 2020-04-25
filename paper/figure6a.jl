using Nystrom, Test, Logging, GeometryTypes, LinearAlgebra, Plots, GmshTools, LaTeXStrings, LinearMaps, IterativeSolvers
using Nystrom: SingleLayerKernel, DoubleLayerKernel, Laplace, Stokes, quadgen, SingleLayerPotential, DoubleLayerPotential,
    Helmholtz, Stokes, Maxwell, SingleLayerOperator, DoubleLayerOperator, IOpCorrection, AdjointDoubleLayerKernel,
    HypersingularKernel, AdjointDoubleLayerOperator, HypersingularOperator

include("../src/Utils/exactsolutions.jl")

function compute_error(op,Γ,qorder,niter,dim=2)
    gmres_iter = Int[]
    nn     = Int[]
    ee     = Float64[]
    for iter in 1:niter
        quad        = quadgen(Γ,qorder)
        npts = length(quad.nodes)
        global nodes_per_element = quad.nodes_per_element
        push!(nn,npts)
        println("DOF: $(npts)")
        println("nodes per element: $(quad.nodes_per_element)")
        nsources    = 3*quad.nodes_per_element
        sources     = dim == 2 ? Nystrom.circle_sources(nsources) : Nystrom.sphere_sources_lebedev(nsources)
        println("number of basis for interpolation: $(length(sources))")
        SL_kernel   = SingleLayerKernel{dim}(op)
        DL_kernel   = DoubleLayerKernel{dim}(op)
        SL_op       = SingleLayerOperator(SL_kernel,quad)
        DL_op       = DoubleLayerOperator(DL_kernel,quad)
        SL_matrix   = Matrix(SL_op)
        DL_matrix   = Matrix(DL_op)
        SLDL_corr        = IOpCorrection(SL_op,SL_matrix,DL_matrix,sources);
        # trace of exact solution
        θ = rand()*2π
        kx = cos(θ)*op.k
        ky = sin(θ)*op.k
        γ₀U         = [-exp(im*(kx*y[1] + ky*y[2]))  for  y in quad.nodes]
        SL1(u)      = SL_matrix*u  - SLDL_corr(u,0,-1)
        DL1(u)      = DL_matrix*u  - SLDL_corr(u,1,0)
        # error in trace
        η = op.k
        L = LinearMap(σ -> σ/2 + DL1(σ) - im*η*SL1(σ),npts)
        rhs = γ₀U
        σ,ch = gmres(L,rhs,verbose=true,log=true,tol=1e-10)
        ua(x) = DoubleLayerPotential(DL_kernel,quad,σ)(x) - im*η*SingleLayerPotential(SL_kernel,quad,σ)(x)
        Rnear = 2
        xtest = [Point(Rnear*cos(θ),Rnear*sin(θ)) for θ in 0:0.1:2π]
        ue(x) = circle_helmholtz_soundsoft(x;R=1,θin=θ,k=op.k)
        Rnear = 2
        ss = 0:0.1:2π |> collect
        xtest = [Point(Rnear*cos(θ),Rnear*sin(θ)) for θ in ss]
        Ue = [ue(x) for x in xtest]
        er1 = [ua(x) - ue(x) for x in xtest]
        push!(ee,norm(er1,Inf)/norm(Ue,Inf))
        push!(gmres_iter,ch.iters)
        println("Trace error:")
        println("$op $dim $(norm(er1,Inf))")
        # error in derivative of greens identity
        Nystrom.refine!(Γ)
    end
    return nn,ee,gmres_iter
end

function fig_gen()
    niter = 8
    fig = plot()
    for qorder = (2,3,4)
        order = qorder+1.
        OPERATORS = (Helmholtz(2π),)
        for  op in OPERATORS
            Γ = Nystrom.circle()
            for _=1:1; Nystrom.refine!(Γ); end
            nn, ee, gmres_iter = compute_error(op,Γ,qorder,niter)
            op isa Laplace ? (opname = "Laplace") : (opname = "Helmholtz")
            plot!(fig,nn,ee,marker=:x,label="M=$qorder  "*opname)
            if op==last(OPERATORS)
                plot!(fig,nn,nn.^(-order)/nn[end]^(-order)*ee[end],xscale=:log10,yscale=:log10,
                      label="order $(Int(order)) slope",xlabel=L"$N_p$",ylabel=L"|error|_\infty",linewidth=4)
            end
        end
    end
    return fig
end

fig = fig_gen()
fname = "/home/lfaria/Dropbox/Luiz-Carlos/general_regularization/figures/fig6a"
savefig(fig,fname)
