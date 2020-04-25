function assemble(I::IntegralOperator)
    I₀ = _assemble_far(I)
    δI = _assemble_near(I)
    return axpy!(1,δI,I₀)
end

_assemble_far(iop::IntegralOperator)  = Matrix(iop)

function _assemble_near_greens(I::IntegralOperator)
    xs        = generate_source_points(iop)
    basis     = [y->SingleLayerKernel(op)(x,y) for x in xs]
    γ₁_basis  = [(y,ny)->DoubleLayerKernel(op)(x,y,ny) for x in xs]
    δI        = GreensCorrection(I,basis,γ₁_basis)
    return δI
end

function generate_source_points(I)
    nodes = getnodes(I.Y)
    bbox  = bounding_box(nodes)
    xc,d  = center(bbox), diameter(bbox)
    generate_source_points(I;center=xc,radius=d/2)
end

function generate_source_points(I;center,radius)
    N = 
end
