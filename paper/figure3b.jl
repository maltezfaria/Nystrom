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
        ADL_kernel   = AdjointDoubleLayerKernel{dim}(op)
        HS_kernel   = HypersingularKernel{dim}(op)
        ADL_op      = AdjointDoubleLayerOperator(ADL_kernel,quad)
        HS_op       = HypersingularOperator(HS_kernel,quad)
        ADL_matrix   = Matrix(ADL_op)
        HS_matrix   = Matrix(HS_op)
        ADLH_corr        = IOpCorrection(ADL_op,ADL_matrix,HS_matrix,sources);
        # exact solution as linear combination of rows of Greens tensor
        xₛ          = 3*ones(Point{dim})
        # trace of exact solution
        T           = eltype(SL_kernel)
        T <: Number ? c = one(T) : c = ones(size(T)[1])
        γ₀U         = [transpose(SL_kernel(xₛ,y))*c   for  y in quad.nodes]
        γ₁U         = [transpose(DL_kernel(xₛ,y,ny))*c for  (y,ny) in zip(quad.nodes,quad.normals)];
        ADL1(u)      = ADL_matrix*u  - ADLH_corr(u,0,-1)
        HS1(u)      =  HS_matrix*u  -  ADLH_corr(u,1,0)
        # error in trace
        L = LinearMap(du -> -du/2 + ADL1(du),npts)
        rhs = HS1(γ₀U)
        du,ch = gmres(L,rhs,verbose=true,log=true,tol=1e-12)
        er1 = γ₁U - du
        push!(ee,norm(er1,Inf)/norm(γ₁U,Inf))
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
        OPERATORS = (Helmholtz(1),)
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
fname = "/home/lfaria/Dropbox/Luiz-Carlos/general_regularization/figures/fig3b.svg"
savefig(fig,fname)
