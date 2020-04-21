module Nystrom

using GeometryTypes: Point, HyperRectangle

################################################################################
## GEOMETRY
################################################################################
include("Geometry/parametricsurface.jl")
include("Geometry/geometry.jl")
include("Geometry/parametricshapes.jl")
# include("Geometry/gmsh.jl")

################################################################################
## QUADRATURE
################################################################################
# include("Quadrature/quadrature.jl")

######################################## GMSH ###################################
# include("GMSH/utils.jl")

end # module
