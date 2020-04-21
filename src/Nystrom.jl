module Nystrom

using GeometryTypes: Point, Normal, HyperRectangle

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

######################################## GMSH ###################################
# include("GMSH/utils.jl")

end # module
