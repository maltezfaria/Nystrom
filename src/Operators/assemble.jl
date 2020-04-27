function assemble(iop::IntegralOperator,::Type{Matrix};correction=GreensCorrection)
    δI = correction(iop)
    I  = Matrix(iop)
    return I + δI
end

# function Base.Matrix(iop::IntegralOperator;correction=GreensCorrection)
#     δI = GreensCorrection(iop)
#     I  = Matrix(iop)
#     return I + δI
# end
