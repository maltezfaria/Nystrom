using Nystrom, Test, Logging, GeometryTypes, LinearAlgebra, Plots, GmshTools, LaTeXStrings, LinearMaps, IterativeSolvers
using Nystrom: SingleLayerKernel, DoubleLayerKernel, Laplace, Stokes, quadgen, SingleLayerPotential, DoubleLayerPotential,
    Helmholtz, Stokes, Maxwell, SingleLayerOperator, DoubleLayerOperator, IOpCorrection, AdjointDoubleLayerKernel,
    HypersingularKernel, AdjointDoubleLayerOperator, HypersingularOperator

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
        θ = π/4
        kx = cos(θ)*op.k
        ky = sin(θ)*op.k
        γ₁U         = [-exp(im*(kx*y[1] + ky*y[2]))*im*(kx*ny[1]+ky*ny[2])   for  (y,ny) in zip(quad.nodes,quad.normals)]
        SL1(u)      = SL_matrix*u  - SLDL_corr(u,0,-1)
        DL1(u)      = DL_matrix*u  - SLDL_corr(u,1,0)
        # error in trace
        L = LinearMap(u -> -u/2 + DL1(u),npts)
        rhs = SL1(γ₁U)
        γ₀U,ch = gmres(L,rhs,verbose=true,log=true,tol=1e-10)
        ua(x) = DoubleLayerPotential(DL_kernel,quad,γ₀U)(x) - SingleLayerPotential(SL_kernel,quad,γ₁U)(x)
        Rnear = 2
        xtest = [Point(Rnear*cos(θ),Rnear*sin(θ)) for θ in 0:0.1:2π]
        ue(x) = Nystrom.circle_helmholtz_soundhard(x;R=1,θin=θ,k=op.k)
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
    for qorder = (1,2,3)
        order = qorder+2.0
        OPERATORS = (Helmholtz(5),)
        for  op in OPERATORS
            Γ = Nystrom.kite()
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
fname = "/home/lfaria/Dropbox/Luiz-Carlos/general_regularization/figures/fig3c.svg"
savefig(fig,fname)
