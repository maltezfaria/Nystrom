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
    xs   = Nystrom.circle_sources(nsources=10,radius=5)
    basis     = [y->SingleLayerKernel(op)(x,y) for x in xs]
    γ₁_basis  = [(y,ny)->DoubleLayerKernel(op)(x,y,ny) for x in xs]
    # kvec      = op.k .* Nystrom.circle_sources(nsources=10,radius=1)
    # basis     = [y->exp(im*dot(k,y)) for k in kvec]
    # γ₁_basis  = [(y,n) -> im*dot(k,n)*exp(im*dot(k,y)) for k in kvec]
    dS   = GreensCorrection(SL,basis,γ₁_basis)
    dD   = GreensCorrection(DL,basis,γ₁_basis)
    dAD   = GreensCorrection(ADL,basis,γ₁_basis)
    dH   = GreensCorrection(H,basis,γ₁_basis)
    ein1 = error_interior_green_identity(SL+dS,DL+dD,γ₀u_in,γ₁u_in)
    ein2 = error_interior_green_identity(ADL+dAD,H+dH,γ₀u_in,γ₁u_in)
    eout1= error_exterior_greens_identity(SL+dS,DL+dD,γ₀u_out,γ₁u_out)
    eout2= error_exterior_greens_identity(ADL,H,γ₀u_out,γ₁u_out)
    return norm(ein1,Inf), norm(ein2,Inf), norm(eout1,Inf), norm(eout2,Inf)
end
