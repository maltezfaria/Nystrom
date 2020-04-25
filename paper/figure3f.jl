using Nystrom, Test, Logging, GeometryTypes, LinearAlgebra, Plots, gmsh, LaTeXStrings, LinearMaps, IterativeSolvers
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
        ADL_kernel   = AdjointDoubleLayerKernel{dim}(op)
        HS_kernel   = HypersingularKernel{dim}(op)
        ADL_op      = AdjointDoubleLayerOperator(ADL_kernel,quad)
        HS_op       = HypersingularOperator(HS_kernel,quad)
        ADL_matrix   = Matrix(ADL_op)
        HS_matrix   = Matrix(HS_op)
        ADLH_corr        = IOpCorrection(ADL_op,ADL_matrix,HS_matrix,sources);
        # trace of exact solution
        θ = π/4
        kx = cos(θ)*op.k
        ky = sin(θ)*op.k
        γ₀U         = [-exp(im*(kx*y[1] + ky*y[2]))   for  y in quad.nodes]
        ADL1(u)      = ADL_matrix*u  - ADLH_corr(u,0,-1)
        HS1(u)      =  HS_matrix*u  -  ADLH_corr(u,1,0)
        # error in trace
        L = LinearMap(du -> du/2 + ADL1(du),npts)
        rhs = HS1(γ₀U)
        du,ch = gmres(L,rhs,verbose=true,log=true,tol=1e-12)
        ua(x) = DoubleLayerPotential(DL_kernel,quad,γ₀U)(x) - SingleLayerPotential(SL_kernel,quad,du)(x)
        Rnear = 2
        xtest = [Point(Rnear*cos(θ),Rnear*sin(θ)) for θ in 0:0.1:2π]
        ue(x) = sphere_helmholtz_soundsoft(x;R=1,θin=θ,k=op.k)
        Rnear = 2
        tt = 0:0.1:π |> collect
        pp = 0:0.1:2π |> collect
        xtest = [Point(Rnear*sin(θ)*cos(ϕ), Rnear*sin(θ)*sin(ϕ),Rnear*cos(θ)) for θ in tt, ϕ in pp]
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
    for qorder = (3,4,5)
        order = qorder-1.0
        OPERATORS = (Helmholtz(5),)
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
fname = "/home/lfaria/Dropbox/Luiz-Carlos/general_regularization/figures/fig3d"
savefig(fig,fname)
