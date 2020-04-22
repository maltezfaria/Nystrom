include("Quadrature/quadrature_test.jl")

using SafeTestsets

@safetestset "Kernel" begin
    using Nystrom, GeometryTypes, FastGaussQuadrature
    op = Laplace()
    SL = SingleLayerKernel()
end
