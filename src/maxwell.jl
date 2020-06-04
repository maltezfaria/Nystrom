############################ Maxwell ############################
struct Maxwell{T1}
    k::T1
end

function (SL::SingleLayerKernel{N,T,Op})(x,y)::T  where {N,T,Op<:Maxwell}
    x==y && return zero(T)
    k = SL.op.k
    r = x .- y
    d = norm(r)
    if N==2
        return error("Maxwell operator not implemented in 2d")
    elseif N==3
        g   = 1/(4π)/d * exp(im*k*d)
        gp  = im*k*g - g/d
        gpp = im*k*gp - gp/d + g/d^2
        ID    = Mat3{Float64}(1,0,0,0,1,0,0,0,1)
        RRT   = r*transpose(r) # r ⊗ rᵗ
        return  g*ID + 1/k^2*(gp/d*ID + (gpp/d^2 - gp/d^3)*RRT)
    end
end
SingleLayerKernel{N}(op::Op,args...) where {N,Op<:Maxwell} = SingleLayerKernel{N,Mat{N,N,ComplexF64,N*N},Op}(op,args...)

# Double Layer Kernel
function (DL::DoubleLayerKernel{N,T,Op})(x,y,ny)::T where {N,T,Op<:Maxwell}
    x==y && return zero(T)
    k = DL.op.k
    r = x .- y
    d = norm(r)
    g   = 1/(4π)/d * exp(im*k*d)
    gp  = im*k*g - g/d
    if N==2
        return error("Maxwell operator not yet defined in 2d")
    elseif N==3
        ID    = Mat3{Float64}(1,0,0,0,1,0,0,0,1)
        ncross = transpose(Mat3{Float64}(0,-ny[3],ny[2],
                                         ny[3],0,-ny[1],
                                         -ny[2],ny[1],0))
        rcross = transpose(Mat3{Float64}(0,-r[3],r[2],
                                         r[3],0,-r[1],
                                         -r[2],r[1],0))
        # return -gp/d*ncross*rcross
        return gp/d*rcross*ncross
    end
end
DoubleLayerKernel{N}(op::Op,args...) where {N,Op<:Maxwell} = DoubleLayerKernel{N,Mat{N,N,ComplexF64,N*N},Op}(op,args...)
