module Nystrom

using GeometryTypes: Point, Normal, HyperRectangle, Mat
using LinearAlgebra
using SpecialFunctions
using RecipesBase
using GmshTools

import ForwardDiff # used for computing jacobian

export
    # shapes
    ellipsis,
    circle,
    kite,
    cube,
    sphere,
    ellipsoid,
    bean,
    refine!,
    # quadrature
    tensorquadrature,
    gmshquadrature,
    # operators
    Laplace,
    Helmholtz,
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
    HyperSingularOperator


################################################################################
## GEOMETRY
################################################################################
include("Geometry/parametricsurface.jl")
include("Geometry/parametricbody.jl")
include("Geometry/parametricshapes.jl")

################################################################################
## QUADRATURE
################################################################################
include("Quadrature/quadrature.jl")

################################################################################
## KERNEL
################################################################################
include("Kernel/kernels.jl")
include("Kernel/laplace.jl")
include("Kernel/helmholtz.jl")
include("Kernel/elastostatic.jl")
include("Kernel/elastodynamic.jl")
# include("Kernel/stokes.jl")
# include("Kernel/maxwell.jl")

################################################################################
## POTENTIALS
################################################################################
include("Potentials/potentials.jl")

################################################################################
## OPERATORS
################################################################################
include("Operators/integraloperators.jl")
include("Operators/corrections.jl")

################################################################################
## UTILS
################################################################################
include("Utils/testutils.jl")
# include("Utils/math.jl")
# include("Utils/conversions.jl")
# include("Utils/utils.jl")


end # module
