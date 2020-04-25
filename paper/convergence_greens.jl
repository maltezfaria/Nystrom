using Nystrom, Test, Logging, GeometryTypes, LinearAlgebra, Plots, gmsh
using Nystrom: SingleLayerKernel, DoubleLayerKernel, Laplace, Stokes, quadgen, SingleLayerPotential, DoubleLayerPotential,
    Helmholtz, Stokes, Maxwell, SingleLayerOperator, DoubleLayerOperator, IOpCorrection, AdjointDoubleLayerKernel,
    HypersingularKernel, AdjointDoubleLayerOperator, HypersingularOperator

dim = 3
op  = Laplace()
# op  = Helmholtz(1)
# op  = Laplace()
# op  = Elastostatic(1,1)
niter = 4
qorder = 3

############################################################
# create a geometry using
if dim == 2
    Γ = Nystrom.ellipsis([1,2])
    Nystrom.refine!(Γ)
elseif dim == 3
    Γ = Nystrom.sphere()
end
############################################################

nn     = Int[]
ee     = Float64[]
ee0 = Float64[]
for iter in 1:niter
    quad        = quadgen(Γ,qorder)
    npts = length(quad.nodes)
    global nodes_per_element = quad.nodes_per_element
    push!(nn,npts)
    println("DOF: $(npts)")
    println("nodes per element: $(quad.nodes_per_element)")
    nsources    = 2*quad.nodes_per_element
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
    T <: Number ? c = ones() : c = ones(size(T)[1])
    γ₀U         = [transpose(SL_kernel(xₛ,y))*c   for  y in quad.nodes]
    γ₁U         = [transpose(DL_kernel(xₛ,y,ny))*c for  (y,ny) in zip(quad.nodes,quad.normals)];
    SL1(u)      = SL_matrix*u  - SLDL_corr(u,0,-1)
    DL1(u)      = DL_matrix*u  - SLDL_corr(u,1,0)
    # error in greens
    er0 = γ₀U/2 - SL_matrix*γ₁U + DL_matrix*γ₀U
    er1 = γ₀U/2 - SL1(γ₁U) + DL1(γ₀U)
    push!(ee0,norm(er0,Inf))
    push!(ee,norm(er1,Inf))
    println("Greens identity error:")
    println("$op $dim $(norm(er0,Inf))")
    println("$op $dim $(norm(er1,Inf))")
    # error in derivative of greens identity
    Nystrom.refine!(Γ)
end

if isa(op,Maxwell)
        order = qorder-1
        order0 = -1
elseif isa(op,Elastostatic)
    order = qorder
    order0 = 0
elseif isa(op,Union{Helmholtz,Laplace})
    if dim == 2
        order = qorder + 1
        order0 = 1
    elseif dim == 3
        order = qorder +1
        order0 = 1
    end
end
hh =  nn .^ (-1/(dim-1))
plot(hh,1 ./ hh.^(-order)*hh[end]^(-order)*ee[end],xscale=:log10,yscale=:log10,label="order=$order",xlabel="h")
plot!(hh,ee,marker=:x,label="greens")
##
plot!(hh,ee0,marker=:x,label="greens")
plot!(hh,1 ./ hh.^(-order0)*hh[end]^(-order0)*ee0[end],xscale=:log10,yscale=:log10,label="order=$order0",xlabel="h")


