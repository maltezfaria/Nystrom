using Nystrom, Test, Logging, GeometryTypes, LinearAlgebra, Plots, gmsh, LaTeXStrings, LinearMaps
using Nystrom: SingleLayerKernel, DoubleLayerKernel, Laplace, Stokes, quadgen, SingleLayerPotential, DoubleLayerPotential,
    Helmholtz, Stokes, Maxwell, SingleLayerOperator, DoubleLayerOperator, IOpCorrection, AdjointDoubleLayerKernel,
    HypersingularKernel, AdjointDoubleLayerOperator, HypersingularOperator

function compute_error(op,quad,r0)
    dim = 2
    npts        = length(quad.nodes)
    nodes_per_element = quad.nodes_per_element
    nsources    = 3*quad.nodes_per_element
    SL_kernel   = SingleLayerKernel{dim}(op)
    DL_kernel   = DoubleLayerKernel{dim}(op)
    SL_op       = SingleLayerOperator(SL_kernel,quad)
    DL_op       = DoubleLayerOperator(DL_kernel,quad)
    SL_matrix   = Matrix(SL_op)
    DL_matrix   = Matrix(DL_op)
    # exact solution as linear combination of rows of Greens tensor
    xₛ          = 3*ones(Point{dim})
    # trace of exact solution
    T           = eltype(SL_kernel)
    T <: Number ? c = ones() : c = ones(size(T)[1])
    γ₀U         = [transpose(SL_kernel(xₛ,y))*c   for  y in quad.nodes]
    γ₁U         = [transpose(DL_kernel(xₛ,y,ny))*c for  (y,ny) in zip(quad.nodes,quad.normals)];

    ee     = Float64[]
    for r in r0
        sources     = Nystrom.circle_sources(nsources,r)
        SLDL_corr        = IOpCorrection(SL_op,SL_matrix,DL_matrix,sources);
        SL1(u)      = SL_matrix*u  - SLDL_corr(u,0,-1)
        DL1(u)      = DL_matrix*u  - SLDL_corr(u,1,0)
        # error in trace
        L = LinearMap(u -> u/2 + DL1(u),npts)
        rhs = SL1(γ₁U)
        du,ch = gmres(L,rhs,verbose=false,log=true,tol=1e-14)
        er1 = γ₀U - du
        push!(ee,norm(er1,Inf)/norm(γ₀U,Inf))
        println("Greens identity error:")
        println("$op $dim $(norm(er1,Inf))")
        println("niter $(ch.iters)")
        println("r0 $(r0)")
    end
    return ee
end

function fig_gen()
    dim = 2
    niter = 4
    op = Helmholtz(1)
    qorder = 5
    Γ = Nystrom.kite()
    for _=1:3; Nystrom.refine!(Γ); end
    fig = plot(xlabel="r",ylabel="error")
    rrange = 2 .^ (1:16)
    for iter = 1:niter
        quad = quadgen(Γ,qorder)
        npts = length(quad.nodes)
        ee = compute_error(op,quad,rrange)
        plot!(fig,rrange,ee,marker=:x,label="N=$npts",xscale=:log10,yscale=:log10)
        Nystrom.refine!(Γ)
    end
    return fig
end

fig = fig_gen()
fname = "/home/lfaria/Dropbox/Luiz-Carlos/general_regularization/figures/fig5a"
savefig(fig,fname)
