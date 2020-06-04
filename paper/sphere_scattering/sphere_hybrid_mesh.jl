using gmsh, Nystrom, FastGaussQuadrature, Plots, LinearAlgebra, IterativeSolvers, Clusters, HMatrices, SparseArrays, LinearMaps
using Nystrom: sphere_helmholtz_soundsoft, Point

function compute_approx_solution(meshsurface,meshoutput,k,p)
    gmsh.initialize()
    pde = Helmholtz(dim=3,k=k)
    ui       = (x) -> exp(im*k*x[1])
    # get
    gmsh.open(meshsurface)
    surface_tags,surface_coords,_ = gmsh.model.mesh.getNodes()
    surface_pts                   = reinterpret(Point{3,Float64},surface_coords) |> collect |> vec
    # generate a quadrature
    dimtags    = gmsh.model.getEntities(2)
    Γ          = quadgengmsh("Gauss$p",dimtags)
    @info "starting computation with $(length(Γ.nodes)) quadrature points..."
    @info "building the cluster trees."
    spl  = GeometricMinimalSplitter(nmax=128)
    clt  = ClusterTree(Γ.nodes,spl)
    bclt = BlockTree(clt,clt)
    permute!(Γ,clt.perm)
    atol = 1e-6
    aca  = HMatrices.PartialACA(atol=atol)
    tsvd = HMatrices.TSVD(atol=atol)
    compress = (x) -> HMatrix(x,bclt,HMatrices.PartialACA(atol=atol))
    # compress = (x) -> HMatrix(x,bclt,(x...) -> HMatrices.compress!(aca(x...),tsvd))
    @info "done."
    @info "Assembling matrices..."
    S,D   = single_double_layer(pde,Γ;compress=compress)
    @info "\t compression rate: S --> $(HMatrices.compression_rate(S)), D --> $(HMatrices.compression_rate(D)),"
    @info "done..."
    @info "solving by gmres..."
    γ₀U   = γ₀(y-> -ui(y),Γ)
    L     = LinearMap(σ -> σ/2 + D*σ - im*k*(S*σ),length(γ₀U))
    σ,ch  = gmres(L,γ₀U,verbose=false,log=true,tol=1e-6,restart=100,maxiter=100)
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
    S,D  = single_double_layer(pde,output_pts,Γ;compress=compress,correction=:greenscorrection)
    @info "\t compression rate: S --> $(HMatrices.compression_rate(S)), D --> $(HMatrices.compression_rate(D)),"
    @info "done..."
    isoutside = [norm(x) >= 1 ? 1 : 0 for x in output_pts]
    Ua        = (D*σ - im*k*(S*σ)).*isoutside
    Ui        = [ui(x) for x in output_pts].*isoutside
    sol_real = vec([ [real(Ua[n] + Ui[n])] for n in 1:length(Ua)])
    sol_imag = vec([ [imag(Ua[n] + Ui[n])] for n in 1:length(Ua)])
    permute!(sol_real,invperm(Xclt.perm))
    permute!(sol_imag,invperm(Xclt.perm))
    view_solution = gmsh.view.add("solution (real part)")
    mname = gmsh.model.getCurrent()
    gmsh.view.addModelData(view_solution,0,mname,"NodeData",output_tags,sol_real)
    gmsh.view.addModelData(view_solution,1,mname,"NodeData",output_tags,sol_imag)
    # write
    fname,ext = split(meshsurface,".")
    out_file = fname*"_p_$(p)_solution.pos"
    gmsh.view.write(view_solution, out_file)
    gmsh.finalize()
    return out_file
end

λ = 0.25
k = 2π/λ
p = 4

meshoutput   = "xyplane.msh"

meshsurface = "sphere_alg_6_order_1_recombine_false.msh"
compute_approx_solution(meshsurface,meshoutput,k,p)
meshsurface = "sphere_alg_6_order_3_recombine_false.msh"
compute_approx_solution(meshsurface,meshoutput,k,p)
meshsurface = "sphere_alg_6_order_1_recombine_true.msh"
compute_approx_solution(meshsurface,meshoutput,k,p)
meshsurface = "sphere_alg_6_order_3_recombine_true.msh"
compute_approx_solution(meshsurface,meshoutput,k,p)
