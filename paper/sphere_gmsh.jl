using Nystrom, FastGaussQuadrature, Plots, LinearAlgebra, IterativeSolvers, Clusters, HMatrices, SparseArrays, LinearMaps, GmshTools
using Nystrom: sphere_helmholtz_soundsoft, Point

function fig_gen(p,h,k)
    gmsh.initialize()
    gmsh.clear()
    pde       = Helmholtz(dim=3,k=k)
    θ         = 0
    ϕ         = 0
    kx = k*sin(θ)*cos(ϕ)
    ky = k*sin(θ)*sin(ϕ)
    kz = k*cos(θ)
    ue(x)     = sphere_helmholtz_soundsoft(x;radius=1,k=k,θin=θ,ϕin = ϕ)
    uᵢ(x) = exp(im*(kx*x[1] + ky*x[2] +  kz*x[3]))
    fname = "/home/lfaria/Dropbox/Luiz-Carlos/general_regularization/draft/figures/sphere.geo"
    gmsh.open(fname)
    gmsh.model.occ.synchronize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.option.setNumber("Mesh.Algorithm",8)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", h)
    gmsh.model.mesh.generate(2)
    gmsh.model.mesh.recombine()
    gmsh.model.mesh.setOrder(p+1) # set it higher than the method's error to avoid geometrical errors
    # extract physical groups
    dict= Dict()
    for (dim,tag) in gmsh.model.getPhysicalGroups()
        name = gmsh.model.getPhysicalName(dim,tag)
        push!(dict,name=>(dim,tag))
    end
    # get nodes on sphere
    dim,tag = dict["Sphere"]
    surface_tags,surface_coords = gmsh.model.mesh.getNodesForPhysicalGroup(dim,tag)
    surface_pts                 = reinterpret(Point{3,Float64},surface_coords) |> collect |> vec
    # generate a quadrature for sphere
    tags    = gmsh.model.getEntitiesForPhysicalGroup(dim,tag)
    dimtags = map(i->(dim,i),tags)
    Γ    = quadgengmsh("Gauss$p",dimtags)
    # get nodes on output surface
    dim,tag = dict["Output"]
    output_tags, coords = gmsh.model.mesh.getNodesForPhysicalGroup(dim,tag)
    coords   = reshape(coords,3,:)[1:3,:]
    X        = reinterpret(Point{3,Float64},coords) |> collect |> vec
    @info "starting computation with $(length(Γ.nodes)) quadrature points..."
    @info "building the cluster trees."
    spl = GeometricMinimalSplitter(nmax=128)
    clt = ClusterTree(Γ.nodes,spl)
    bclt = BlockTree(clt,clt)
    permute!(Γ,clt.perm)
    atol = 1e-6
    compress = (x) -> HMatrix(x,bclt,HMatrices.PartialACA(atol=atol))
    @info "done."
    @info "Assembling matrices..."
    S,D   = single_double_layer(pde,Γ;compress=compress)
    @info "\t compression rate: S --> $(HMatrices.compression_rate(S)), D --> $(HMatrices.compression_rate(D)),"
    @info "done..."
    @info "solving by gmres..."
    γ₀U   = γ₀(y-> -uᵢ(y),Γ)
    L     = LinearMap(σ -> σ/2 + D*σ - im*k*(S*σ),length(γ₀U))
    σ,ch  = gmres(L,γ₀U,verbose=false,log=true,tol=1e-6,restart=100,maxiter=100)
    @info "done. gmres converged in $(ch.iters) iterations."
    @info "Evaluating field error at $(length(X)) observation points"
    spl  = GeometricMinimalSplitter(nmax=128)
    Xclt = ClusterTree(X,spl,reorder=true)
    bclt = BlockTree(Xclt,clt)
    S,D  = single_double_layer(pde,X,Γ;compress=compress,correction=:greenscorrection)
    isoutside = [norm(x)>=1 for x in X]
    Ua   = (D*σ - im*k*(S*σ)).*isoutside
    Ui   = [uᵢ(x) for x in X].*isoutside
    @info "computing exact solution..."
    Ue   = [ue(x) for x in X].*isoutside
    @info "done."
    @show norm(Ue - Ua,Inf)
    sol_real = vec([ [real(Ua[n] + Ui[n])] for n in 1:length(Ua)])
    ee       = vec([ [abs(Ua[n] - Ue[n])] for n in 1:length(Ua)])
    permute!(sol_real,invperm(Xclt.perm))
    permute!(ee,invperm(Xclt.perm))
    view_solution = gmsh.view.add("solution (real part)")
    view_error    = gmsh.view.add("error")
    gmsh.view.addModelData(view_solution,0,"sphere","NodeData",output_tags,sol_real)
    gmsh.view.addModelData(view_error,0,"sphere","NodeData",output_tags,ee)
    # add trace
    trace_real =  vec([ [0] for pt in surface_pts]);
    gmsh.view.addModelData(view_solution,0,"sphere","NodeData",surface_tags,trace_real)
    # write
    gmsh.view.write(view_solution, "/home/lfaria/Dropbox/Luiz-Carlos/general_regularization/draft/figures/sphere_solution.msh")
    gmsh.view.write(view_error, "/home/lfaria/Dropbox/Luiz-Carlos/general_regularization/draft/figures/sphere_error.msh")
    gmsh.finalize()
    return nothing
end

p = 2
λ = 1
k = 2π/λ
h = λ/10
fig = fig_gen(p,h,k)
