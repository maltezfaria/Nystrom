using Nystrom, Test, Logging, GeometryTypes, LinearAlgebra, Plots, GmshTools, LaTeXStrings, LinearMaps, IterativeSolvers
using Nystrom: SingleLayerKernel, DoubleLayerKernel, Laplace, Stokes, quadgen, SingleLayerPotential, DoubleLayerPotential,
    Helmholtz, Stokes, Maxwell, SingleLayerOperator, DoubleLayerOperator, IOpCorrection, AdjointDoubleLayerKernel,
    HypersingularKernel, AdjointDoubleLayerOperator, HypersingularOperator

function compute_error(op,Γ,qorder,dim=2)
    gmres_iter = Int[]
    nn     = Int[]
    ee     = Float64[]
    quad        = quadgen(Γ,qorder)
    npts = length(quad.nodes)
    nodes_per_element = quad.nodes_per_element
    nsources    = 3*quad.nodes_per_element
    sources     = dim == 2 ? Nystrom.circle_sources(nsources) : Nystrom.sphere_sources_lebedev(nsources)
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
    # error in trace
    L = LinearMap(u -> u/2 + DL1(u),npts)
    rhs = SL1(γ₁U)
    du,ch = gmres(L,rhs,verbose=false,log=true,tol=1e-8,restart=1000)
    er1 = γ₀U - du
    println("k = $(op.k):")
    println("npts = $npts")
    println("iter = $(ch.iters):")
    println("error = $(norm(er1,Inf)):")
    # error in derivative of greens identity
    return npts,ch.iters,norm(er1,Inf)
end

function fig_gen()
    fig = plot()
    for qorder = (2,3,4,5)
        Γ = Nystrom.kite()
        for _ in 1:3; Nystrom.refine!(Γ); end
        krange = 2 .^ (1:7)
        nn = Int[]
        ii = Int[]
        ee = Float64[]
        for k = krange
            op = Helmholtz(k)
            npts, iter, er = compute_error(op,Γ,qorder)
            push!(nn,npts)
            push!(ii,iter)
            push!(ee,er)
            Nystrom.refine!(Γ)
        end
        plot!(fig,krange,ii,xlabel="k",ylabel="gmres iterations",marker=:x,
              label="M=$qorder")
    end
    return fig
end

fig = fig_gen()
fname = "/home/lfaria/Dropbox/Luiz-Carlos/general_regularization/figures/fig4a.svg"
savefig(fig,fname)
