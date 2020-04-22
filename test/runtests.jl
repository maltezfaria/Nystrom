include("Quadrature/quadrature_test.jl")

using SafeTestsets

@safetestset "Kernel" begin
    using Nystrom, GeometryTypes, FastGaussQuadrature, LinearAlgebra
    x = rand(Point{2,Float64})
    y = rand(Point{2,Float64})
    op = Laplace(ndims=2)
    op = Helmholtz(ndims=2,k=1)
    op = Elastostatic(ndims=2,μ=1,λ=1)
    op = Elastodynamic(ndims=2,μ=1,λ=1,ω=1,ρ=1.0)
    G = SingleLayerKernel(op)
    G(x,y)
    dG = DoubleLayerKernel(op)
    dGt = AdjointDoubleLayerKernel(op)
    d2G = HyperSingularKernel(op)
    @testset "Potential" begin
        @testset "Helmholtz" begin
            geo  = circle()
            quad = tensorquadrature((20,),geo,gausslegendre)
            op = Helmholtz(ndims=2,k=1)
            𝒮 = IntegralPotential(op,quad,:singlelayer)
            𝒮 = SingleLayerPotential(op,quad)
            𝒟 = DoubleLayerPotential(op,quad)
            σ  = SurfaceDensity(ComplexF64,quad)
            u(x)  = 𝒮[σ](x)
            v(x)  = 𝒟[σ](x)
            @test u(y) == 0
            @test v(y) == 0
            k    = Vec(1,2)
            f(x) = exp(im*(k⋅x))
            σ    = γ₀(f,quad)
            Nystrom.∇(::typeof(f)) = (x)->im.*k.*f(x) #need to define the gradient
            σ    = γ₁(f,quad)
            S = SingleLayerOperator(op,quad,quad)
            D = DoubleLayerOperator(op,quad,quad)
            AD = AdjointDoubleLayerOperator(op,quad,quad)
            AD = AdjointDoubleLayerOperator(op,quad,quad)
        end
    end
end
