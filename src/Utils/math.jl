@inline function vdot(a,b)
    @assert length(a) == length(b)
    return sum( a .* b)
end

function window(t,t0,t1)::Float64
    if abs(t) <= t0
        return 1
    elseif t0 < abs(t) < t1
        u = (abs(t) - t0) / (t1 - t0)
        return exp(2*exp(-1/u)/(u-1))
    else
        return 0
    end
end

function mysqrt(c::Number)
    θ = mod(angle(c),2π)/2
    ρ = sqrt(norm(c))
    return ρ*Complex(cos(θ),sin(θ))
end


