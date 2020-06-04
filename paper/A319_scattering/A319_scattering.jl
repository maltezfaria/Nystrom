using Nystrom, gmsh, GeometryTypes, HMatrices, LinearMaps, IterativeSolvers, LinearAlgebra, Clusters
path = @__DIR__

function main(meshsurface,k,p)
    pde  = Helmholtz(dim=3,k=k)
    fname,ext = split(meshsurface,".")
    gmsh.initialize()
    gmsh.open(meshsurface)
    tags, coords, _ = gmsh.model.mesh.getNodes(2,-1,true)
    pts = reshape(coords,3,:)[1:3,:]
    pts = reinterpret(Point3{Float64},pts);

    dimtags = gmsh.model.getEntities(2)
    Γ    = quadgengmsh("Gauss$p",dimtags)
    npts = length(Γ.nodes)
    @info npts

    spl  = GeometricMinimalSplitter(nmax=128)
    clt  = ClusterTree(Γ.nodes,spl)
    bclt = BlockTree(clt,clt)
    permute!(Γ,clt.perm)
    rtol = 1e-6
    aca  = HMatrices.PartialACA(rtol=rtol)
    tsvd = HMatrices.TSVD(rtol=rtol)
    # compress = (x) -> HMatrix(x,bclt,HMatrices.PartialACA(rtol=rtol))
    compress = (x) -> HMatrix(x,bclt,(x...) -> HMatrices.compress!(aca(x...),tsvd))
    S,D   = single_double_layer(pde,Γ;compress=compress)
    @info "\t compression rate: S --> $(HMatrices.compression_rate(S)), D --> $(HMatrices.compression_rate(D)),"
    @info "solving by gmres..."
    # solve the BIE
    ui(x) = exp(im*k*x[1])
    γ₀U         = γ₀((y) -> -ui(y), Γ)
    η = k
    L = LinearMap(σ -> σ/2 + D*σ - im*η*(S*σ),npts)
    rhs = γ₀U
    σ,ch = gmres(L,rhs,verbose=true,log=true,tol=1e-4,restart=300,maxiter=300)
    # export trace
    sol_real = vec([ [0] for n in 1:length(pts)])
    # sol_imag = vec([ [imag(σ[n])] for n in 1:length(σ)])
    view_solution = gmsh.view.add("trace")
    mname = gmsh.model.getCurrent()
    gmsh.view.addModelData(view_solution,0,mname,"NodeData",tags,sol_real)
    # gmsh.view.addModelData(view_solution,1,"A319","NodeData",tags,sol_imag)
    out_file = fname*"_trace.pos"
    gmsh.view.write(view_solution, out_file)
    # export solution on xy plane
    S,D = nothing,nothing
    Base.GC.gc()
    gmsh.open("xyplane.msh")
    tags, coords, _ = gmsh.model.mesh.getNodes(2,-1,true)
    pts = reshape(coords,3,:)[1:3,:]
    X = [Point{3,Float64}(pts[:,n]) for n=1:size(pts,2)]
    Xclt  = ClusterTree(X,spl,reorder=true)
    bclt  = BlockTree(Xclt,clt)
    compress = (x) -> HMatrix(x,bclt,HMatrices.PartialACA(rtol=rtol))
    @info "Creating potential..."
    S,D   = single_double_layer(pde,X,Γ;compress=compress,correction=:nothing)
    @info "done."
    uinc       = [ui(pt) for pt in X]
    uscat      = D*σ - im*k*(S*σ)
    sol_real = vec([ [real(uscat[n] + uinc[n])] for n in 1:length(uscat)])
    sol_imag = vec([ [imag(uscat[n] + uinc[n])] for n in 1:length(uscat)])
    permute!(sol_real,invperm(Xclt.perm))
    permute!(sol_imag,invperm(Xclt.perm))
    view_solution = gmsh.view.add("solution xyplane")
    mname = gmsh.model.getCurrent()
    gmsh.view.addModelData(view_solution,0,mname,"NodeData",tags,sol_real)
    gmsh.view.addModelData(view_solution,1,mname,"NodeData",tags,sol_imag)
    out_file = fname*"_xyplane.pos"
    gmsh.view.write(view_solution, out_file)
    # export solution on xzplane
    gmsh.open("xzplane.msh")
    tags, coords, _ = gmsh.model.mesh.getNodes(2,-1,true)
    pts = reshape(coords,3,:)[1:3,:]
    X = [Point{3,Float64}(pts[:,n]) for n=1:size(pts,2)]
    Xclt  = ClusterTree(X,spl,reorder=true)
    bclt  = BlockTree(Xclt,clt)
    compress = (x) -> HMatrix(x,bclt,HMatrices.PartialACA(rtol=rtol))
    @info "Creating potential..."
    S,D   = single_double_layer(pde,X,Γ;compress=compress,correction=:nothing)
    @info "done."
    uinc       = [ui(pt) for pt in X]
    uscat      = D*σ - im*k*(S*σ)
    sol_real = vec([ [real(uscat[n] + uinc[n])] for n in 1:length(uscat)])
    sol_imag = vec([ [imag(uscat[n] + uinc[n])] for n in 1:length(uscat)])
    permute!(sol_real,invperm(Xclt.perm))
    permute!(sol_imag,invperm(Xclt.perm))
    view_solution = gmsh.view.add("solution xzplane")
    mname = gmsh.model.getCurrent()
    gmsh.view.addModelData(view_solution,0,mname,"NodeData",tags,sol_real)
    gmsh.view.addModelData(view_solution,1,mname,"NodeData",tags,sol_imag)
    out_file = fname*"_xzplane.pos"
    gmsh.view.write(view_solution, out_file)
    gmsh.finalize()
    return out_file,ch
end

λ = 2e3
k = 2π/λ
p = 2

mesh = "A319_order_1_recombine_false.msh"
# mesh = "A319_refined.msh"
outfile,ch = main(mesh,k,p)
