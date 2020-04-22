############################ ELASTODYNAMIC ############################3
struct Elastodynamic{T}
    μ::T
    λ::T
    ω::T
    ρ::T
end
# Single Layer
function (SL::SingleLayerKernel{N,T,Op})(x,y)::T  where {N,T,Op<:Elastodynamic}
    x==y && return zero(T)
    μ = SL.op.μ
    λ = SL.op.λ
    ω = SL.op.ω
    ρ = SL.op.ρ
    c1 = sqrt((λ + 2μ)/ρ)
    c2 = sqrt(μ/ρ)
    r = x-y
    d = norm(r)
    RRT = r*transpose(r) # r ⊗ rᵗ
    s = -im*ω
    z1 = s*d/c1
    z2 = s*d/c2
    if N==2
        α = 2
        ID = Mat2{Float64}(1,0,0,1)
        ψ = (-c2/c1 * besselk(1,z1) + besselk(1,z2))/z2 + besselk(0,z2)
        chi = -(c2/c1)^2*(besselk(0,z1) + 2/z1*besselk(1,z1)) + besselk(0,z2) + 2/z2*besselk(1,z2)
    elseif N==3
        α = 4
        ID    = Mat3{Float64}(1,0,0,0,1,0,0,0,1)
        ψ     = exp(-z2)/d + (1+z2)/(z2^2)*exp(-z2)/d - c2^2/c1^2*(1+z1)/(z1^2)*exp(-z1)/d
        chi   = 3*ψ - 2*exp(-z2)/d - c2^2/c1^2*exp(-z1)/d
    end
    return 1/(α*π*μ)*(ψ*ID - chi*RRT/d^2)
end
SingleLayerKernel{N}(op::Op,args...) where {N,Op<:Elastodynamic} = SingleLayerKernel{N,Mat{N,N,Complex{Float64},N*N},Op}(op,args...)

# Double Layer Kernel
function (DL::DoubleLayerKernel{N,T,Op})(x,y,ny)::T where {N,T,Op<:Elastodynamic}
    x==y && return zero(T)
    μ = DL.op.μ
    λ = DL.op.λ
    ω = DL.op.ω
    ρ = DL.op.ρ
    c1 = sqrt((λ + 2μ)/ρ)
    c2 = sqrt(μ/ρ)
    r = x-y
    d = norm(r)
    RRT = r*transpose(r) # r ⊗ rᵗ
    drdn = vdot(r,ny)/d
    s = -im*ω
    z1 = s*d/c1
    z2 = s*d/c2
    if N==2
        ID = Mat2{Float64}(1,0,0,1)
        α = 2
        ##
        ψ = (-c2/c1 * besselk(1,z1) + besselk(1,z2))/z2 + besselk(0,z2)
        ##
        ψr = 1/(d*z2)* (c2/c1 * (besselk(0,z1)*z1 + 2*besselk(1,z1)) -
                        (besselk(0,z2)*z2 + 2*besselk(1,z2)) -
                        besselk(1,z2)*z2^2
                        )
        ##
        chi = -(c2/c1)^2*(besselk(0,z1) + 2/z1*besselk(1,z1)) + besselk(0,z2) + 2/z2*besselk(1,z2)
        ##
        chir = 1/d*( c2^2/c1^2*(besselk(1,z1)*z1 + 2*(besselk(0,z1) + 2/z1*besselk(1,z1))) -
                   (besselk(1,z2)*z2 + 2*(besselk(0,z2) + 2/z2*besselk(1,z2)))
                  )
    elseif N==3
        α = 4
        ID = Mat3{Float64}(1,0,0,0,1,0,0,0,1)
        ψ    = exp(-z2)/d + (1+z2)/(z2^2)*exp(-z2)/d - c2^2/c1^2*(1+z1)/(z1^2)*exp(-z1)/d
        chi  = 3*ψ - 2*exp(-z2)/d - c2^2/c1^2*exp(-z1)/d
        ψr   = -chi/d - (1+z2)*exp(-z2)/d^2
        chir = -3chi/d - (1+z2)*exp(-z2)/d^2 + c2^2/c1^2*(1+z1)*exp(-z1)/d^2
    end
    return 1/(α*π) * ( (ψr - chi/d) * (drdn*ID + ny*transpose(r)/d) -
                      2*chi/d*(r*transpose(ny)/d - 2*drdn*RRT/d^2) - 2*chir*drdn*RRT/d^2 +
                      (c1^2/c2^2 -2)*(ψr - chir - α/2*chi/d)*r*transpose(ny)/d
                    )
end
DoubleLayerKernel{N}(op::Op,args...) where {N,Op<:Elastodynamic} = DoubleLayerKernel{N,Mat{N,N,Complex{Float64},N*N},Op}(op,args...)

# Adjoint Double Layer Kernel
function (ADL::AdjointDoubleLayerKernel{N,T,Op})(x,y,nx)::T where {N,T,Op<:Elastodynamic}
    x==y && return zero(T)
    μ = ADL.op.μ
    λ = ADL.op.λ
    ω = ADL.op.ω
    ρ = ADL.op.ρ
    c1 = sqrt((λ + 2μ)/ρ)
    c2 = sqrt(μ/ρ)
    r = x-y
    d = norm(r)
    RRT = r*transpose(r) # r ⊗ rᵗ
    RNXT = r*transpose(nx)
    NXRT = nx*transpose(r)
    drdn = vdot(r,nx)/d
    s = -im*ω
    z1 = s*d/c1
    z2 = s*d/c2
    if N==2
        ID = Mat2{Float64}(1,0,0,1)
        α = 2
        ##
        ψ = (-c2/c1 * besselk(1,z1) + besselk(1,z2))/z2 + besselk(0,z2)
        ##
        ψr = 1/(d*z2)* (c2/c1 * (besselk(0,z1)*z1 + 2*besselk(1,z1)) -
                        (besselk(0,z2)*z2 + 2*besselk(1,z2)) -
                        besselk(1,z2)*z2^2
                        )
        ##
        chi = -(c2/c1)^2*(besselk(0,z1) + 2/z1*besselk(1,z1)) + besselk(0,z2) + 2/z2*besselk(1,z2)
        ##
        chir = 1/d*( c2^2/c1^2*(besselk(1,z1)*z1 + 2*(besselk(0,z1) + 2/z1*besselk(1,z1))) -
                   (besselk(1,z2)*z2 + 2*(besselk(0,z2) + 2/z2*besselk(1,z2)))
                  )
    elseif N==3
        α = 4
        ID = Mat3{Float64}(1,0,0,0,1,0,0,0,1)
        ψ    = exp(-z2)/d + (1+z2)/(z2^2)*exp(-z2)/d - c2^2/c1^2*(1+z1)/(z1^2)*exp(-z1)/d
        chi  = 3*ψ - 2*exp(-z2)/d - c2^2/c1^2*exp(-z1)/d
        ψr   = -chi/d - (1+z2)*exp(-z2)/d^2
        chir = -3chi/d - (1+z2)*exp(-z2)/d^2 + c2^2/c1^2*(1+z1)*exp(-z1)/d^2
    end
    return 1/(α*π) * ( 2*(chir - 2*chi/d)*dot(nx,r)*RRT/d^3 + 2*chi/d^2*NXRT -
                       (ψr - chi/d)*(dot(r,nx)*ID/d + RNXT/d)  -
                       λ/μ*( ψr - chir - α/2*chi/d)*NXRT/d
                       )
end
AdjointDoubleLayerKernel{N}(op::Op,args...) where {N,Op<:Elastodynamic} = AdjointDoubleLayerKernel{N,Mat{N,N,Complex{Float64},N*N},Op}(op,args...)

# Hypersingular kernel
function (HS::HypersingularKernel{N,T,Op})(x,y,nx,ny)::T where {N,T,Op<:Elastodynamic}
    x==y && return zero(T)
    μ = HS.op.μ
    λ = HS.op.λ
    ω = HS.op.ω
    ρ = HS.op.ρ
    c1 = sqrt((λ + 2μ)/ρ)
    c2 = sqrt(μ/ρ)
    r = x-y
    d = norm(r)
    RRT = r*transpose(r) # r ⊗ rᵗ
    drdn = vdot(r,ny)/d
    s = -im*ω
    z1 = s*d/c1
    z2 = s*d/c2
    if N==2
        ID = Mat2{Float64}(1,0,0,1)
        α = 2
        ##
        ψ = (-c2/c1 * besselk(1,z1) + besselk(1,z2))/z2 + besselk(0,z2)
        ##
        ψr = 1/(d*z2)* (c2/c1 * (besselk(0,z1)*z1 + 2*besselk(1,z1)) -
                        (besselk(0,z2)*z2 + 2*besselk(1,z2)) -
                        besselk(1,z2)*z2^2
                        )
        ψrr = (-c2/c1 * (3*besselk(0,z1)*z1 + besselk(1,z1)*(z1^2+6)) +
           (3*besselk(0,z2)*z2 + besselk(1,z2)*(z2^2+6)) + (besselk(0,z2)*z2 + besselk(1,z2))*z2^2
           )*1/z2/d^2
        ##
        chi = -(c2/c1)^2*(besselk(0,z1) + 2/z1*besselk(1,z1)) + besselk(0,z2) + 2/z2*besselk(1,z2)
        ##
        chir = 1/d*( c2^2/c1^2*(besselk(1,z1)*z1 + 2*(besselk(0,z1) + 2/z1*besselk(1,z1))) -
                     (besselk(1,z2)*z2 + 2*(besselk(0,z2) + 2/z2*besselk(1,z2)))
                     )
        chirr =  1/d^2*(-(c2/c1)^2*(besselk(0,z1)*z1^2 + 3*besselk(1,z1)*z1 + 6*(besselk(0,z1) + 2/z1*besselk(1,z1)))+
                    (besselk(0,z2)*z2^2 + 3*besselk(1,z2)*z2 + 6*(besselk(0,z2) + 2/z2*besselk(1,z2))))
    elseif N==3
        α = 4
        ID = Mat3{Float64}(1,0,0,0,1,0,0,0,1)
        ψ    = exp(-z2)/d + (1+z2)/(z2^2)*exp(-z2)/d - c2^2/c1^2*(1+z1)/(z1^2)*exp(-z1)/d
        chi  = 3*ψ - 2*exp(-z2)/d - c2^2/c1^2*exp(-z1)/d
        ψr   = -chi/d - (1+z2)*exp(-z2)/d^2
        ψrr  = 4*chi/d^2 + (3 + 3*z2 + z2^2)*exp(-z2)/d^3 - c2^2/c1^2*(1+z1)*exp(-z1)/d^3
        chir = -3chi/d - (1+z2)*exp(-z2)/d^2 + c2^2/c1^2*(1+z1)*exp(-z1)/d^2
        chirr = 12*chi/d^2 + (5+5*z2 + z2^2)*exp(-z2)/d^3 - c2^2/c1^2*(5+5*z1+z1^2)*exp(-z1)/d^3
    end
    return μ/(α*π) * (drdn* ( 4*(chirr - 5*chir/d + 8*chi/d^2)*dot(r,nx)*RRT/d^3
                              - (ψrr - ψr/d - 3*chir/d + 6*chi/d^2) *
                              (r*transpose(nx)/d + dot(r,nx)/d*ID) + 2*(2*chir/d
                              - 4*chi/d^2 + λ/μ*(chirr + (α-2)/2*chir/d -
                              α*chi/d^2 - ψrr + ψr/d))*nx*transpose(r)/d ) + 2 *
                              (2*chir/d - 4*chi/d^2 + λ/μ * (chirr +
                              (α-2)/2*chir/d - α*chi/d^2 - ψrr + ψr/d)) *
                              dot(r,nx)*r*transpose(ny)/d^2 - (ψrr - ψr/d -
                              3*chir/d + 6*chi/d^2)*(dot(nx,ny)*RRT +
                              dot(r,nx)*ny*transpose(r))/d^2 + (4*chi/d^2 +
                              λ/μ*4*(chir/d + α/2*chi/d^2 - ψr/d) +
                              λ^2/μ^2*(chirr + α*chir/d + (α-2)*chi/d^2 - ψrr
                              - α/2*ψr/d))*nx*transpose(ny) - 2*(ψr/d -
                              chi/d^2)*(dot(nx,ny)*ID + ny*transpose(nx)) )
end
HypersingularKernel{N}(op::Op,args...) where {N,Op<:Elastodynamic} = HypersingularKernel{N,Mat{N,N,Complex{Float64},N*N},Op}(op,args...)