using SafeTestsets

@safetestset "Greens formula" begin
    using Nystrom, LinearAlgebra, GeometryTypes, FastGaussQuadrature
    using Nystrom: error_greens_identity
    geo = circle()
    op  = Helmholtz(dim=2,k=1)
    # op  = Elastostatic(dim=2,μ=1,λ=1)
    xin, xout = Point(0.2,-0.1), Point(5.2,-3.)
    @testset "No correction" begin
        # create a geometry
        for n=1:4
            refine!(geo)
            quad = tensorquadrature((10,),geo,gausslegendre)
            @show error_greens_identity(op,quad,xin,xout)
        end
    end
    @testset "Greens correction" begin
        geo = circle()
        for _=1:5; refine!(geo); end
        op = Laplace(dim=2)
        p  = 4
        quad = tensorquadrature((p,),geo,gausslegendre)
        S    = SingleLayerOperator(op,quad)
        D    = DoubleLayerOperator(op,quad)
        xs   = Nystrom.circle_sources(nsources=10,radius=5)
        basis     = [y->SingleLayerKernel(op)(x,y) for x in xs]
        γ₁_basis  = [(y,ny)->DoubleLayerKernel(op)(x,y,ny) for x in xs]
        dS   = GreensCorrection(S,basis,γ₁_basis)
        # x    = SurfaceDensity(ComplexF64,quad)
        # dS(x)
    end
    @testset "Greens correction near" begin
        h = 1e-10
        bdy1 = circle(radius=0.5,center=(0,0))
        bdy2  = circle(radius=0.5,center=(1+h,0))
        for _=1:5; refine!(bdy1); refine!(bdy2); end
        op = Laplace(dim=2)
        p  = 4
        quad = tensorquadrature((p,),[bdy1,bdy2],gausslegendre)
        Xquad = tensorquadrature((p,),bdy1,gausslegendre)
        Yquad = tensorquadrature((p,),bdy2,gausslegendre)
        S    = SingleLayerOperator(op,Xquad,Yquad)
        D    = DoubleLayerOperator(op,Xquad,Yquad)
        xs   = Nystrom.circle_sources(nsources=10,center=(1+h,0),radius=5)
        basis     = [y->SingleLayerKernel(op)(x,y) for x in xs]
        γ₁_basis  = [(y,ny)->DoubleLayerKernel(op)(x,y,ny) for x in xs]
        dS   = GreensCorrection(S,basis,γ₁_basis)
        # x    = SurfaceDensity(ComplexF64,quad)
        # dS(x)
    end
end


