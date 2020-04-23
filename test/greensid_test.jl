using SafeTestsets

@safetestset "Greens formula" begin
    using Nystrom, LinearAlgebra, GeometryTypes, FastGaussQuadrature
    using Nystrom: error_greens_identity
    # create a geometry
    geo = circle()
    op = Laplace(ndims=2)
    xin, xout = Point(0.2,-0.1), Point(5.2,-3.)
    for n=1:4
        refine!(geo)
        quad = tensorquadrature((10,),geo,gausslegendre)
        @show error_greens_identity(op,quad,xin,xout)
    end
end


