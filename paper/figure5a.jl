using Nystrom, FastGaussQuadrature, Plots, LinearAlgebra, IterativeSolvers, Clusters, HMatrices, SparseArrays, LinearMaps, GmshTools
using Nystrom: sphere_helmholtz_soundsoft, Point, @gmsh

const mesh = gmsh.model.mesh

function compute_approx_solution(meshsurface,meshoutput,k,p)
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
    compress = (x) -> HMatrix(x,bclt,(x...) -> aca(x...))
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
    S,D  = single_double_layer(pde,output_pts,Γ;compress=compress,correction=:greenscorrection)
    isoutside = [norm(x) >= 1 ? 1 : 0 for x in output_pts]
    Ua        = (D*σ - im*k*(S*σ)).*isoutside
    Ui        = [ui(x) for x in output_pts].*isoutside
    sol_real = vec([ [real(Ua[n] + Ui[n])] for n in 1:length(Ua)])
    permute!(sol_real,invperm(Xclt.perm))
    view_solution = gmsh.view.add("solution (real part)")
    mname = gmsh.model.getCurrent()
    gmsh.view.addModelData(view_solution,0,mname,"NodeData",output_tags,sol_real)
    # write
    fname,ext = split(meshsurface,".")
    gmsh.view.write(view_solution, fname*"_solution."*ext)
    return nothing
end

gmsh.initialize()

λ = 0.5
k = 2π/λ
p = 2

meshoutput   = "xyplane.msh"
meshsurface = "sphere_alg_6_order_1_recombine_false.msh"
compute_approx_solution(meshsurface,meshoutput,k,p)
meshsurface = "sphere_alg_6_order_3_recombine_false.msh"
compute_approx_solution(meshsurface,meshoutput,k,p)
meshsurface = "sphere_alg_6_order_1_recombine_true.msh"
compute_approx_solution(meshsurface,meshoutput,k,p)
meshsurface = "sphere_alg_6_order_3_recombine_true.msh"
compute_approx_solution(meshsurface,meshoutput,k,p)

gmsh.finalize()

# p = 2
# λ = 1
# pde = Helmholtz(dim=3,k=2π/λ)
# path = "/home/lfaria/Dropbox/Luiz-Carlos/general_regularization/draft/figures/"
# mesh_file = joinpath(path,"sphere_with_planes_fulltri.msh")
# fig = fig_gen(pde,mesh_file,p)
# mesh_file = joinpath(path,"sphere_with_planes_fullquad.msh")
# fig = fig_gen(pde,mesh_file,p)
# mesh_file = joinpath(path,"sphere_with_planes_hybrid.msh")
# fig = fig_gen(pde,mesh_file,p)
