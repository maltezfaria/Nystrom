using Nystrom, Test, Logging, GeometryTypes, LinearAlgebra, Plots, GmshTools, LaTeXStrings
using Nystrom: SingleLayerKernel, DoubleLayerKernel, quadgen, SingleLayerPotential, DoubleLayerPotential,
    Helmholtz, Stokes, Maxwell, SingleLayerOperator, DoubleLayerOperator, IOpCorrection, AdjointDoubleLayerKernel,
    HypersingularKernel, AdjointDoubleLayerOperator, HypersingularOperator

function compute_error(op,Γ,qorder,niter,dim=2)
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
        # sources     = dim == 2 ? Nystrom.circle_sources(nsources) : Nystrom.sphere_sources_uniform(nsources)
        sources     = dim == 2 ? Nystrom.circle_sources(nsources) : Nystrom.sphere_sources_lebedev(nsources)
        println("number of basis for interpolation: $(length(sources))")
        SL_kernel   = SingleLayerKernel{dim}(op)
        DL_kernel   = DoubleLayerKernel{dim}(op)
        SL_op       = SingleLayerOperator(SL_kernel,quad)
        DL_op       = DoubleLayerOperator(DL_kernel,quad)
        SL_matrix   = Matrix(SL_op)
        DL_matrix   = Matrix(DL_op)
        SLDL_corr        = IOpCorrection(SL_op,SL_matrix,DL_matrix,sources);
        # exact solution as linear combination of rows of Greens tensor
        xₛ          = 3*ones(Point{dim})
        # trace of exact solution
        T           = eltype(SL_kernel)
        T <: Number ? c = one(T) : c = ones(size(T)[1])
        γ₀U         = [transpose(SL_kernel(xₛ,y))*c   for  y in quad.nodes]
        γ₁U         = [transpose(DL_kernel(xₛ,y,ny))*c for  (y,ny) in zip(quad.nodes,quad.normals)];
        SL1(u)      = SL_matrix*u  - SLDL_corr(u,0,-1)
        DL1(u)      = DL_matrix*u  - SLDL_corr(u,1,0)
        # error in greens
        er0 = γ₀U/2 - SL_matrix*γ₁U + DL_matrix*γ₀U
        er1 = γ₀U/2 - SL1(γ₁U) + DL1(γ₀U)
        @info norm(DL1(γ₀U))
        @info norm(SL1(γ₁U))
        push!(ee,norm(er1,Inf))
        println("Greens identity error:")
        println("$op $dim $(norm(er0,Inf))")
        println("$op $dim $(norm(er1,Inf))")
        # error in derivative of greens identity
        Nystrom.refine!(Γ)
    end
    return nn,ee
end

function fig_gen()
    dim = 2
    niter = 5
    fig = plot()
    for qorder = (3,4,5)
        order = qorder+0.0
        OPERATORS = (Elastostatic(1,1), Elastodynamic(1,1,1,1))
        for  op in OPERATORS
            Γ = Nystrom.kite()
            Nystrom.refine!(Γ)
            nn, ee = compute_error(op,Γ,qorder,niter)
            op isa Elastostatic ? (opname = "Elastostatic") : (opname = "Elastodynamic")
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
fname = "/home/lfaria/Dropbox/Luiz-Carlos/general_regularization/figures/fig1b.svg"
savefig(fig,fname)
fig
