using Nystrom, gmsh GeometryTypes, HMatrices, LinearMaps, IterativeSolvers, LinearAlgebra, Clusters
path = @__DIR__

function main()

    geo_order = 1
    quad_order = 3
    λ = 100.0e3 # units in mm
    k = 2π/λ
    h = λ/5

    gmsh.initialize()
    gmsh.clear()

    mname = "airplane"
    gmsh.model.add(mname)
    gmsh.option.setNumber("General.Terminal", 1)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", h);
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", h);
    gmsh.option.setNumber("Mesh.ElementOrder", geo_order);
    gmsh.merge(joinpath(path,"A319.brep"))
    gmsh.model.list()
    gmsh.model.mesh.generate(2)
    gmsh.write(joinpath(path,"A319_meshsize_$h.msh"))
    # gmsh.merge("A319_meshsize_$h.msh")
    tags, coords, _ = gmsh.model.mesh.getNodes(2,-1,true)
    pts = reshape(coords,3,:)[1:3,:]
    pts = reinterpret(Point3{Float64},pts);
    sol_real =  vec([ [0] for pt in pts]);
    view_solution = gmsh.view.add("solution (real part)")
    gmsh.view.addModelData(view_solution,0,mname,"NodeData",tags,sol_real)
    gmsh.view.write(view_solution, "airplane.msh")

    Γ = Nystrom.quadgengmsh("Gauss$quad_order",2)
    npts = length(Γ.nodes)
    @info npts

    clt = ClusterTree(Γ.nodes)
    permute!(Γ,clt.perm)
    @assert Γ.nodes == clt.data
    iperm = invperm(clt.perm)
    bclt        = BlockTree(clt,clt)

    pde  = Helmholtz(dim=3,k=k)
    ker  = Nystrom.SingleLayerKernel(pde)
    SL_op       = SingleLayerOperator(pde,Γ)
    DL_op       = DoubleLayerOperator(pde,Γ)
    # SL_full   = Matrix(SL_op)
    # DL_full   = Matrix(DL_op)
    println("building single layer...")
    compression = HMatrices.PartialACA(atol=1e-8)
    SL_matrix   = HMatrix(SL_op,bclt,compression)
    println("done.")
    println("building double layer...")
    DL_matrix   = HMatrix(DL_op,bclt,compression)
    δS = GreensCorrection(SL_op,SL_matrix,DL_matrix)
    println("building sparse correction")
    nsources    = 2*quad.nodes_per_element
    println("building interpolant...")
    SLDL_corr        = Nystrom.IOpCorrection(SL_op,SL_matrix,DL_matrix,sources,iperm);
    println("done.")

# trace of exact solution
    θ = π/2
    ϕ = 0
    kx = op.k*sin(θ)*cos(ϕ)
    ky = op.k*sin(θ)*sin(ϕ)
    kz = op.k*cos(θ)
    γ₀U         = [-exp(im*(kx*y[1] + ky*y[2] +  kz*y[3]))  for  y in quad.nodes]
    SL1(u)      = SL_matrix*(w.*u)  - SLDL_corr(u,0,-1,iperm)
    DL1(u)      = DL_matrix*(w.*u)  - SLDL_corr(u,1,0,iperm)
    # DL1(u)      = DL_matrix*(w.*u)
    # error in trace
    η = op.k
    L = LinearMap(σ -> σ/2 + DL1(σ) - im*η*SL1(σ),npts)
    # L = LinearMap(σ -> σ/2 + DL1(σ),npts)
    rhs = γ₀U
    sigma,ch = gmres(L,rhs,verbose=true,log=true,tol=1e-5,restart=2000,maxiter=2000)

    #export solution
    uᵢ(x) = exp(im*(kx*x[1] + ky*x[2] +  kz*x[3]))
    mname = "xy-plane"
    gmsh.model.add(mname)
    h = λ/10;
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", h);
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", h);
    lx = -2e4; wx = 8e4
    ly = -2e4; wy = 4e4
    rec = gmsh.model.occ.addRectangle(lx,ly,0,wx,wy)
    rec2 = gmsh.model.occ.addRectangle(lx,ly,0,wx,wy)
    gmsh.model.occ.rotate([(2,rec2)],0,1,0,1,0,0,π/2)
    gmsh.model.occ.synchronize()
    gmsh.model.mesh.generate(2)
    tags, coords, _ = gmsh.model.mesh.getNodes(2,-1,true)
    pts = reshape(coords,3,:)[1:3,:]
    X = [Point{3,Float64}(pts[:,n]) for n=1:size(pts,2)]

    SL_pot       = SingleLayerPotential(SL_kernel,X,quad)
    DL_pot       = DoubleLayerPotential(DL_kernel,X,quad)

    kern(i,j)::ComplexF64 = DL_pot[i,j] - im*η*SL_pot[i,j]

    Xclt,permX = ClusterTree(X)
    hoptions   = HMatrixOptions(1e-5,HMatrices.ACA_partial,false,false,true)
    bclt       = BlockClusterTree(Xclt,clt)
    @info "Creating potential..."
    Pot = build_hmatrix(kern,bclt,hoptions)
    @info "done."
    uinc       = [uᵢ(pt) for pt in X]
    uscat      = Pot*(w.*sigma)
    sol_real = vec([ [real(uscat[n] + uinc[n])] for n in 1:length(uscat)])
    permute!(sol_real,invperm(permX))

    view_solution = gmsh.view.add("solution (real part)")
    gmsh.view.addModelData(view_solution,0,mname,"NodeData",tags,sol_real)
    gmsh.view.write(view_solution, "xy_yz_planes.msh")

    gmsh.finalize()

end

main()
