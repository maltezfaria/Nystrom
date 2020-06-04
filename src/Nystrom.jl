module Nystrom

using GeometryTypes: Point, Normal, Mat
using LinearAlgebra
using Base.Threads
using SpecialFunctions
using RecipesBase
using DoubleFloats
using Suppressor
using IterativeSolvers
using SparseArrays
using GSL
using GmshTools
using gmsh
using Printf

export
    # operators
    Laplace,
    Helmholtz,
    HelmholtzPML,
    Elastostatic,
    Elastodynamic,
    Stokes,
    Maxwell,
    # kernels
    SingleLayerKernel,
    DoubleLayerKernel,
    AdjointDoubleLayerKernel,
    HyperSingularKernel,
    # density
    SurfaceDensity,
    # potentials
    SingleLayerPotential,
    DoubleLayerPotential,
    IntegralPotential,
    # trace
    γ₀,
    γ₁,
    # integral operators
    IntegralOperator,
    SingleLayerOperator,
    DoubleLayerOperator,
    AdjointDoubleLayerOperator,
    HyperSingularOperator,
    single_double_layer,
    singlelayer,
    doublelayer,
    adjointdoublelayer_hypersingular,
    # corrections to integral operators
    GreensCorrection,
    # quadrature
    quadgengmsh

include("interface.jl")
include("gmshquadrature.jl")
include("kernels.jl")
include("laplace.jl")
include("helmholtz.jl")
include("elastostatic.jl")
include("elastodynamic.jl")
include("stokes.jl")
include("potential.jl")
include("integraloperators.jl")
include("greenscorrection.jl")
include("assemble.jl")
include("geometryutils.jl")
include("mathutils.jl")
include("conversions.jl")
include("exactsolutions.jl")
# include("Utils/gmshaddons.jl")
# include("Utils/utils.jl")

function debug(flag=true)
    if flag
        @eval ENV["JULIA_DEBUG"] = "Nystrom"
    end
end

end # module
