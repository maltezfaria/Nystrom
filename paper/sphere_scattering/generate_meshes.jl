using gmsh, LinearAlgebra
using Nystrom: Point, sphere_helmholtz_soundsoft

gmsh.initialize()

gmsh.option.setNumber("General.Terminal",1)

function generate_sphere_mesh(alg,order,h,recombine)
    gmsh.clear()
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax",h)
    geo = gmsh.model.occ.addSphere(0,0,0,1)
    gmsh.model.occ.synchronize()
    # set options
    gmsh.model.mesh.setAlgorithm(2,geo,alg)
    recombine && gmsh.model.mesh.setRecombine(2,geo)
    # generate and write
    gmsh.model.mesh.generate(2)
    gmsh.model.mesh.setOrder(order)
    fname = "sphere_alg_$(alg)_order_$(order)_recombine_$(recombine).msh"
    gmsh.write(fname)
end

λ = 0.25
k = 2π/λ

gmsh.option.setNumber("Mesh.RecombinationAlgorithm",0)
# generate a mesh of the xyplane for outputting the solution
gmsh.model.add("xyplane")
rec = gmsh.model.occ.addRectangle(-3,-3,0,6,6,0)
gmsh.model.occ.synchronize()
gmsh.model.mesh.setTransfiniteSurface(rec)
gmsh.model.mesh.setRecombine(2,rec)
N = 6*10/λ
for (dim,tag) in gmsh.model.getBoundary((2,rec))
    gmsh.model.mesh.setTransfiniteCurve(tag,N)
end
gmsh.model.mesh.generate(2)
gmsh.write("xyplane.msh")

generate_sphere_mesh(6,1,λ/3,false)
generate_sphere_mesh(6,1,λ/3,true)
generate_sphere_mesh(6,3,λ/2,false)
generate_sphere_mesh(6,3,λ/3,true)

function compute_exact_solution(mesh,k)
    gmsh.open(mesh)
    ui       = (x) -> exp(im*k*x[1])
    mname = gmsh.model.getCurrent()
    output_tags, coords, _ = gmsh.model.mesh.getNodes()
    coords   = reshape(coords,3,:)[1:3,:]
    X        = reinterpret(Point{3,Float64},coords) |> collect |> vec
    isoutside = [norm(x) >= 1 for x in X]
    us = (x) -> sphere_helmholtz_soundsoft(x;radius=1,k=k,θin=π/2,ϕin = 0)
    # compute exact solution
    Ue = Vector{ComplexF64}(undef,length(X))
    Threads.@threads for i=1:length(X)
        x = X[i]
        Ue[i] =(us(x)+ui(x)).*isoutside[i]
    end
    # export it to gmsh
    solution_real      = vec([ [real(Ue[n])] for n in 1:length(Ue)])
    solution_imag      = vec([ [imag(Ue[n])] for n in 1:length(Ue)])
    view_solution = gmsh.view.add("exact solution")
    gmsh.view.addModelData(view_solution,0,mname,"NodeData",output_tags,solution_real)
    gmsh.view.addModelData(view_solution,1,mname,"NodeData",output_tags,solution_imag)
    gmsh.view.write(view_solution, "exact_solution.msh")
    return nothing
end

@info "computing exact solution..."
# compute_exact_solution("xyplane.msh",k)
@info "done"

gmsh.finalize()
