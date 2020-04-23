function error_greens_identity(op,mesh,xin,xout)
    SL  = SingleLayerOperator(op,mesh)
    DL  = DoubleLayerOperator(op,mesh)
    ADL = AdjointDoubleLayerOperator(op,mesh)
    H   = HyperSingularOperator(op,mesh)
    # construct an exact interior solution by taking a source point outside
    u         = (x)   -> SingleLayerKernel(op)(xout,x)
    dudn      = (x,n) -> DoubleLayerKernel(op)(xout,x,n)
    γ₀u_in    = γ₀(u,mesh)
    γ₁u_in    = γ₁(dudn,mesh)
    # construct exact exterior solution by taking a source point inside
    u         = (x) -> SingleLayerKernel(op)(xin,x)
    dudn      = (x,n) -> DoubleLayerKernel(op)(xin,x,n)
    γ₀u_out        = γ₀(u,mesh)
    γ₁u_out        = γ₁(dudn,mesh)
    # compute the errors
    ein1 = error_interior_greens_identity(SL,DL,γ₀u_in,γ₁u_in)
    ein2 = error_interior_greens_identity(ADL,H,γ₀u_in,γ₁u_in)
    eout1= error_exterior_greens_identity(SL,DL,γ₀u_out,γ₁u_out)
    eout2= error_exterior_greens_identity(ADL,H,γ₀u_out,γ₁u_out)
    return norm(ein1,Inf), norm(ein2,Inf), norm(eout1,Inf), norm(eout2,Inf)
end

error_interior_greens_identity(SL,DL,γ₀u,γ₁u) = γ₀u/2 - SL*γ₁u + DL*γ₀u
error_exterior_greens_identity(SL,DL,γ₀u,γ₁u) = -γ₀u/2 - SL*γ₁u + DL*γ₀u
