using Nystrom, Test, Logging, GeometryTypes, LinearAlgebra, Plots, LaTeXStrings, LinearMaps, IterativeSolvers, GmshTools
using Nystrom: SingleLayerKernel, DoubleLayerKernel, Laplace, Stokes, quadgen, SingleLayerPotential, DoubleLayerPotential,
    Helmholtz, Stokes, Maxwell, SingleLayerOperator, DoubleLayerOperator, IOpCorrection, AdjointDoubleLayerKernel,
    HypersingularKernel, AdjointDoubleLayerOperator, HypersingularOperator

include("../src/Utils/exactsolutions.jl")

function compute_error(op,Γ,qorder,niter,dim=3)
    gmres_iter = Int[]
    nn     = Int[]
    ee     = Float64[]
    for iter in 1:niter
        quad        = quadgen(Γ,qorder)
        npts = length(quad.nodes)
        nodes_per_element = quad.nodes_per_element
        push!(nn,npts)
        println("DOF: $(npts)")
        println("nodes per element: $(quad.nodes_per_element)")
        nsources    = 2*quad.nodes_per_element
        sources     = Nystrom.sphere_sources_lebedev(nsources)
        println("number of basis for interpolation: $(length(sources))")
        SL_kernel   = SingleLayerKernel{dim}(op)
        DL_kernel   = DoubleLayerKernel{dim}(op)
        SL_op       = SingleLayerOperator(SL_kernel,quad)
        DL_op       = DoubleLayerOperator(DL_kernel,quad)
        SL_matrix   = Matrix(SL_op)
        DL_matrix   = Matrix(DL_op)
        SLDL_corr        = IOpCorrection(SL_op,SL_matrix,DL_matrix,sources);
        # trace of exact solution
        θ = 0
        ϕ = 0
        kx = op.k*sin(θ)*cos(ϕ)
        ky = op.k*sin(θ)*sin(ϕ)
        kz = op.k*cos(θ)
        γ₀U         = [-exp(im*(kx*y[1] + ky*y[2] +  kz*y[3]))  for  y in quad.nodes]
        SL1(u)      = SL_matrix*u  - SLDL_corr(u,0,-1)
        DL1(u)      = DL_matrix*u  - SLDL_corr(u,1,0)
        # error in trace
        η = op.k
        L = LinearMap(σ -> σ/2 + DL1(σ) - im*η*SL1(σ),npts)
        rhs = γ₀U
        σ,ch = gmres(L,rhs,verbose=true,log=true,tol=1e-5)
        ua(x) = DoubleLayerPotential(DL_kernel,quad,σ)(x) - im*η*SingleLayerPotential(SL_kernel,quad,σ)(x)
        Rnear = 2
        xtest = [Point(Rnear*sin(θ)*cos(ϕ),Rnear*sin(θ)*sin(ϕ),Rnear*cos(θ)) for θ in 0:0.1:π, ϕ in 0:0.1:2π]
        ue(x) = sphere_helmholtz_soundsoft(x;R=1,k=op.k,θin=θ,ϕin = ϕ)
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
    niter = 4
    fig = plot()
    for qorder = (1,2,3)
        order = qorder+1.0
        OPERATORS = (Helmholtz(5),)
        for  op in OPERATORS
            Γ = Nystrom.sphere()
            for _=1:1; Nystrom.refine!(Γ); end
            nn, ee, gmres_iter = compute_error(op,Γ,qorder,niter)
            nn = sqrt.(nn)
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
fname = "/home/lfaria/Dropbox/Luiz-Carlos/general_regularization/figures/fig6b"
savefig(fig,fname)
