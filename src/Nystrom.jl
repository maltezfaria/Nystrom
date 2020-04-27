module Nystrom

using GeometryTypes: Point, Normal, Mat
using LinearAlgebra
using SpecialFunctions
using RecipesBase
using GmshTools

import ForwardDiff # used for computing jacobian

export
    # geometry
    ellipsis,
    circle,
    kite,
    cube,
    sphere,
    ellipsoid,
    bean,
    refine!,
    domain,
    quadgen!,
    meshgen!,
    tensorquadrature,
    gmshquadrature,
    Domain,
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
    HyperSingularOperator,
    # corrections to integral operators
    GreensCorrection


################################################################################
## GEOMETRY
################################################################################
include("Geometry/cuboid.jl")
include("Geometry/parametricsurface.jl")
include("Geometry/parametricbody.jl")
include("Geometry/parametricshapes.jl")
include("Geometry/quadrature.jl")
include("Geometry/domain.jl")

################################################################################
## QUADRATURE
################################################################################
# include("Quadrature/quadrature.jl")

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
include("Utils/geometryutils.jl")
# include("Utils/math.jl")
# include("Utils/conversions.jl")
# include("Utils/utils.jl")


end # module
