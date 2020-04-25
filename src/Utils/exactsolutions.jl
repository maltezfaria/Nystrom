using SpecialFunctions

function circle_helmholtz_soundsoft(xobs;R=1,k=1,θin=0)
    x = xobs[1]
    y = xobs[2]
    r = sqrt(x^2+y^2)
    θ = atan(y,x)
    u = 0.0
    r < R && return u
    c(n) = -exp(im*n*(π/2-θin))*besselj(n,k*R)/besselh(n,k*R)
    u    = c(0)*besselh(0,k*r)
    n = 1;
    while (abs(c(n)) > 1e-12)
        u += c(n)*besselh(n,k*r)*exp(im*n*θ) + c(-n)*besselh(-n,k*r)*exp(-im*n*θ)
        n += 1
    end
    return u
end

function circle_helmholtz_soundhard(xobs;R=1,k=1,θin=0)
    x = xobs[1]
    y = xobs[2]
    r = sqrt(x^2+y^2)
    θ = atan(y,x)
    u = 0.0
    r < R && return u
    besseljp(n,r) = (besselj(n-1,r) - besselj(n+1,r))/2 #derivative of besselj
    besselhp(n,r) = (besselh(n-1,r) - besselh(n+1,r))/2 #derivative of besselh
    c(n) = -exp(im*n*(π/2-θin))*besseljp(n,k*R)/besselhp(n,k*R)
    u    = c(0)*besselh(0,k*r)
    n = 1;
    while (abs(c(n)) > 1e-12)
        u += c(n)*besselh(n,k*r)*exp(im*n*θ) + c(-n)*besselh(-n,k*r)*exp(-im*n*θ)
        n += 1
    end
    return u
end


sphbesselj(l,r) = sqrt(π/(2r)) * besselj(l+1/2,r)
sphbesselh(l,r) = sqrt(π/(2r)) * besselh(l+1/2,r)
# should not be called with m<0 nor with |x|>1. Uses GNU scientific library
function sphharmonic(l,m,θ,ϕ)
    out = ccall((:gsl_sf_legendre_sphPlm,"libgsl"), Cdouble, (Cint,Cint,Cdouble), l, abs(m), cos(θ))*exp(im*m*ϕ)
    if m>0
        return out
    else
        return (-1^m)*out
    end
end

function sphere_helmholtz_soundsoft(xobs;R=1,k=1,θin=0,ϕin=0)
    x = xobs[1]
    y = xobs[2]
    z = xobs[3]
    r = sqrt(x^2+y^2+z^2)
    θ = acos(z/r)
    ϕ = atan(y,x)
    u = 0.0
    r < R && return u
    c(l,m) = -4π*im^l*sphharmonic(l,-m,θin,ϕin)*sphbesselj(l,k*R)/sphbesselh(l,k*R)
    u    = 0
    l = 0
    while (maximum(abs(c(l,m)) for m=-l:l) > 1e-12)
    # for l=0:9
        for m=-l:l
            u += c(l,m)*sphbesselh(l,k*r)*sphharmonic(l,m,θ,ϕ)
        end
        l += 1
    end
    return u
end

