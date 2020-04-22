module Nystrom

using GeometryTypes: Point, Normal, HyperRectangle
using LinearAlgebra
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
    # quadrature
    tensorquadrature,
    gmshquadrature,
    # operators
    Laplace,
    Helmholtz,
    Elastostatic,
    Elastodynamic,
    Stokes,
    Maxwell

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
include("Kernel/stokes.jl")
include("Kernel/maxwell.jl")

################################################################################
## POTENTIALS
################################################################################
include("Potentials/potentials.jl")

################################################################################
## OPERATORS
################################################################################
# include("Operators/integraloperators.jl")
# include("Operators/corrections.jl")

################################################################################
## UTILS
################################################################################
# include("Utils/math.jl")
# include("Utils/conversions.jl")
# include("Utils/utils.jl")


end # module
