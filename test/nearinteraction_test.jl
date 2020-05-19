using SafeTestsets

@safetestset "Near interaction" begin
    using Nystrom, LinearAlgebra, GeometryTypes, FastGaussQuadrature
    using Nystrom: near_interaction_list_bodies
    Γ   = Domain{2}()
    xc  = Point(0,0.)
    δ   = 0.1
    c1  = circle()
    c2  = circle(center=(2+δ,0.))
    push!(Γ,c1,c2)
    meshgen!(Γ,0.1)
    quadgen!(Γ,(3,),algo1d=gausslegendre)
    plot(Γ)
    quad = Γ.quadrature
    list = near_interaction_list_bodies(quad,quad;tol=0.2)
end


