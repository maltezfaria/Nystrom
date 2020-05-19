module Nystrom

using GeometryTypes: Point, Normal, Mat
using LinearAlgebra
using SpecialFunctions
using RecipesBase
using DoubleFloats
using Suppressor
using IterativeSolvers
using SparseArrays
using GSL
using GmshTools
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

################################################################################
## KERNEL
################################################################################
include("Kernel/kernels.jl")
include("Kernel/laplace.jl")
include("Kernel/helmholtz.jl")
include("Kernel/elastostatic.jl")
include("Kernel/elastodynamic.jl")
include("Kernel/stokes.jl")
# include("Kernel/maxwell.jl")

################################################################################
## POTENTIALS
################################################################################
include("Potentials/potentials.jl")

################################################################################
## OPERATORS
################################################################################
include("Operators/integraloperators.jl")
include("Operators/greenscorrection.jl")
include("Operators/assemble.jl")

################################################################################
## UTILS
################################################################################
include("Utils/testutils.jl")
include("Utils/geometryutils.jl")
include("Utils/math.jl")
include("Utils/conversions.jl")
include("Utils/exactsolutions.jl")
include("Utils/gmshaddons.jl")
# include("Utils/utils.jl")


end # module
