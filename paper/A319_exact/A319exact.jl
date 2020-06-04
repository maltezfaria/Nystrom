using gmsh, Nystrom, FastGaussQuadrature, Plots, LinearAlgebra, IterativeSolvers, Clusters, HMatrices, SparseArrays, LinearMaps
using Nystrom: sphere_helmholtz_soundsoft, Point

function compute_approx_solution(meshsurface,meshoutput,k,p)
    gmsh.initialize()
    pde = Helmholtz(dim=3,k=k)
    xs  = Point(2e4,0,0.)
    ue       = (x) -> 1e5*SingleLayerKernel(pde)(xs,x)
    # get
    gmsh.open(meshsurface)
    surface_tags,surface_coords,_ = gmsh.model.mesh.getNodes()
    surface_pts                   = reinterpret(Point{3,Float64},surface_coords) |> collect |> vec
    # generate a quadrature
    dimtags    = gmsh.model.getEntities(2)
    Γ          = quadgengmsh("Gauss$p",dimtags)
    @show "surface area: $(sum(Γ.weights))"
    @info "starting computation with $(length(Γ.nodes)) quadrature points..."
    @info "building the cluster trees."
    spl  = GeometricMinimalSplitter(nmax=128)
    clt  = ClusterTree(Γ.nodes,spl,reorder=false)
    bclt = BlockTree(clt,clt)
    permute!(Γ,clt.perm)
    atol = 1e-6
    aca  = HMatrices.PartialACA(atol=atol)
    tsvd = HMatrices.TSVD(atol=atol)
    # compress = (x) -> HMatrix(x,bclt,HMatrices.PartialACA(atol=atol))
    compress = (x) -> HMatrix(x,bclt,(x...) -> HMatrices.compress!(aca(x...),tsvd))
    @info "done."
    @info "Assembling matrices..."
    S,D   = single_double_layer(pde,Γ;compress=compress)
    @info "\t compression rate: S --> $(HMatrices.compression_rate(S)), D --> $(HMatrices.compression_rate(D)),"
    @info "done..."
    @info "solving by gmres..."
    γ₀U   = γ₀(y-> ue(y),Γ)
    L     = LinearMap(σ -> σ/2 + D*σ - im*k*(S*σ),length(γ₀U))
    σ,ch  = gmres(L,γ₀U,verbose=false,log=true,tol=1e-4,restart=1000,maxiter=1000)
    @info "done. gmres converged in $(ch.iters) iterations."
    gmsh.open(meshoutput)
    output_tags,output_coords,_ = gmsh.model.mesh.getNodes()
    output_pts                  = reinterpret(Point{3,Float64},output_coords) |> collect |> vec
    @info "Evaluating field error at $(length(output_pts)) observation points"
    spl  = GeometricMinimalSplitter(nmax=128)
    Xclt = ClusterTree(output_pts,spl,reorder=true)
    bclt = BlockTree(Xclt,clt)
    compress = (x) -> HMatrix(x,bclt,HMatrices.PartialACA(atol=atol))
    # compress = (x) -> HMatrix(x,bclt,(x...) -> HMatrices.compress!(aca(x...),tsvd))
    @info "Assembling matrices..."
    S,D = nothing,nothing
    Base.GC.gc()
    S,D  = single_double_layer(pde,output_pts,Γ;compress=compress,correction=:nothing)
    @info "\t compression rate: S --> $(HMatrices.compression_rate(S)), D --> $(HMatrices.compression_rate(D)),"
    @info "done..."
    Ua        = (D*σ - im*k*(S*σ))
    Ue        = [ue(x) for x in output_pts]
    error_real = vec([ [real(Ua[n] - Ue[n])] for n in 1:length(Ua)])
    error_imag = vec([ [imag(Ua[n] - Ue[n])] for n in 1:length(Ua)])
    UeMax = norm(Ue,Inf)
    error_abs = vec([ [abs(Ua[n] - Ue[n])/UeMax] for n in 1:length(Ua)])
    permute!(error_real,invperm(Xclt.perm))
    permute!(error_imag,invperm(Xclt.perm))
    permute!(error_abs,invperm(Xclt.perm))
    view_solution = gmsh.view.add("error qorder $(p)")
    mname = gmsh.model.getCurrent()
    gmsh.view.addModelData(view_solution,0,mname,"NodeData",output_tags,error_real)
    gmsh.view.addModelData(view_solution,1,mname,"NodeData",output_tags,error_imag)
    gmsh.view.addModelData(view_solution,2,mname,"NodeData",output_tags,error_abs)
    # write
    fname,ext = split(meshsurface,".")
    out_file = fname*"_p_$(p)_solution.pos"
    gmsh.view.write(view_solution, out_file)
    @show ch.iters
    gmsh.finalize()
    return out_file, ch
end

λ = 2e3
k = 2π/λ

meshoutput   = "output.msh"

# meshsurface = "A319_order_1_recombine_false_h_200.msh"
# outfile,ch = compute_approx_solution(meshsurface,meshoutput,k,1)
# outputfile,ch = compute_approx_solution(meshsurface,meshoutput,k,2)
# meshsurface = "A319_order_1_recombine_false_h_400.msh"
# outfile, ch = compute_approx_solution(meshsurface,meshoutput,k,1)
# outputfile,ch = compute_approx_solution(meshsurface,meshoutput,k,2)
# outputfile,ch = compute_approx_solution(meshsurface,meshoutput,k,4)
meshsurface = "A319_order_1_recombine_false_h_100.msh"
outfile, ch = compute_approx_solution(meshsurface,meshoutput,k,1)
# meshsurface = "A319_order_1_recombine_true.msh"
# compute_approx_solution(meshsurface,meshoutput,k,1)
# compute_approx_solution(meshsurface,meshoutput,k,2)
