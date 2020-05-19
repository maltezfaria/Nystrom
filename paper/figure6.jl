using Nystrom, GmshTools, GeometryTypes, HMatrices, LinearMaps, IterativeSolvers, LinearAlgebra, Clusters
path = @__DIR__

function main()
    geo_order = 1
    quad_order = 1
    λ = 5.0e3 # units in mm
    k = 2π/λ
    h = λ/10
    pde  = Helmholtz(dim=3,k=k)

    gmsh.initialize()
    gmsh.clear()
    mname = "airplane"
    gmsh.model.add(mname)
    gmsh.option.setNumber("General.Terminal", 1)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", h);
    gmsh.option.setNumber("Mesh.ElementOrder", geo_order);
    gmsh.merge(joinpath(path,"A319.brep"))
    gmsh.model.list()
    gmsh.model.mesh.generate(2)
    gmsh.model.mesh.setOrder(geo_order)
    gmsh.write(joinpath(path,"A319_meshsize_$h.msh"))
    # gmsh.merge("A319_meshsize_$h.msh")
    tags, coords, _ = gmsh.model.mesh.getNodes(2,-1,true)
    pts = reshape(coords,3,:)[1:3,:]
    pts = reinterpret(Point3{Float64},pts);
    sol_real =  vec([ [0] for pt in pts]);
    view_solution = gmsh.view.add("solution (real part)")
    gmsh.view.addModelData(view_solution,0,mname,"NodeData",tags,sol_real)
    gmsh.view.write(view_solution, "airplane.msh")

    dimtags = gmsh.model.getEntities(2)
    Γ    = quadgengmsh("Gauss$quad_order",dimtags)
    npts = length(Γ.nodes)
    @info npts

    spl  = GeometricMinimalSplitter(nmax=128)
    clt  = ClusterTree(Γ.nodes,spl)
    bclt = BlockTree(clt,clt)
    permute!(Γ,clt.perm)
    atol = 1e-6
    compress = (x) -> HMatrix(x,bclt,HMatrices.PartialACA(atol=atol))
    println("building matrices...")
    S,D   = single_double_layer(pde,Γ;compress=compress)
    println("done.")
    @info "\t compression rate: S --> $(HMatrices.compression_rate(S)), D --> $(HMatrices.compression_rate(D)),"
    @info "done..."
    @info "solving by gmres..."

# trace of exact solution
    θ = 0
    ϕ = 0
    kx = k*sin(θ)*cos(ϕ)
    ky = k*sin(θ)*sin(ϕ)
    kz = k*cos(θ)
    γ₀U         = [-exp(im*(kx*y[1] + ky*y[2] +  kz*y[3]))  for  y in Γ.nodes]
    η = k
    L = LinearMap(σ -> σ/2 + D*σ - im*η*(S*σ),npts)
    rhs = γ₀U
    σ,ch = gmres(L,rhs,verbose=true,log=true,tol=10*atol,restart=2000,maxiter=200)
    #export solution
    uᵢ(x) = exp(im*(kx*x[1] + ky*x[2] +  kz*x[3]))
    mname = "xy-plane"
    gmsh.model.add(mname)
    h = λ/10;
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
    Xclt  = ClusterTree(X,spl,reorder=true)
    bclt  = BlockTree(Xclt,clt)
    compress = (x) -> HMatrix(x,bclt,HMatrices.PartialACA(atol=atol))
    @info "Creating potential..."
    S,D   = single_double_layer(pde,X,Γ;compress=compress)
    @info "done."
    uinc       = [uᵢ(pt) for pt in X]
    uscat      = D*σ - im*k*(S*σ)
    sol_real = vec([ [real(uscat[n] + uinc[n])] for n in 1:length(uscat)])
    permute!(sol_real,invperm(Xclt.perm))

    view_solution = gmsh.view.add("solution (real part)")
    gmsh.view.addModelData(view_solution,0,mname,"NodeData",tags,sol_real)
    gmsh.view.write(view_solution, "A319_xy_yz_planes.msh")

    gmsh.finalize()

end

main()
