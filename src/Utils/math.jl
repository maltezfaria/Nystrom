@inline function vdot(a,b)
    @assert length(a) == length(b)
    return sum( a .* b)
end

function window(t,t0,t1)
    if abs(t) <= t0
        return 1
    elseif t0 < abs(t) < t1
        u = (abs(t) - t0) / (t1 - t0)
        return exp(2*exp(-1/u)/(u-1))
    else
        return 0
    end
end

function complex_scaling(t,a,θ)
    if abs(t) <= a
        t
    elseif t < -a
        # t  + im*(t+a)*sin(θ)
        -a + (t+a)*exp(im*θ)
    else # t>a
        # t  + im*(t-a)*sin(θ)
        a + (t-a)*exp(im*θ)
    end
end

function derivative_complex_scaling(t,a,θ)
    if abs(t) <= a
        ComplexF64(1)
    else
        exp(im*θ)
    end
end
