using GmshTools, LinearAlgebra
using Nystrom: Point, sphere_helmholtz_soundsoft

gmsh.initialize()

function generate_sphere_mesh(alg,order,h,recombine)
    gmsh.clear()
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
h = λ/2.5 # approximate element size
k = 2π/λ

gmsh.option.setNumber("Mesh.RecombinationAlgorithm",0)
gmsh.option.setNumber("Mesh.CharacteristicLengthMax",h)
# generate a mesh of the xyplane for outputting the solution
gmsh.model.add("xyplane")
rec = gmsh.model.occ.addRectangle(-3,-3,0,6,6,0)
gmsh.model.occ.synchronize()
gmsh.model.mesh.setTransfiniteSurface(rec)
gmsh.model.mesh.setRecombine(2,rec)
N = 12*10/λ
for (dim,tag) in gmsh.model.getBoundary((2,rec))
    gmsh.model.mesh.setTransfiniteCurve(tag,N)
end
gmsh.model.mesh.generate(2)
gmsh.write("xyplane.msh")

generate_sphere_mesh(6,1,h,false)
generate_sphere_mesh(6,1,h,true)
generate_sphere_mesh(6,3,h,false)
generate_sphere_mesh(6,3,h,true)

function compute_exact_solution(mesh,k)
    gmsh.open(mesh)
    ui       = (x) -> exp(im*k*x[1])
    mname = gmsh.model.getCurrent()
    output_tags, coords, _ = gmsh.model.mesh.getNodes()
    coords   = reshape(coords,3,:)[1:3,:]
    X        = reinterpret(Point{3,Float64},coords) |> collect |> vec
    isoutside = [norm(x) >= 1 for x in X]
    us = (x) -> sphere_helmholtz_soundsoft(x;radius=1,k=k,θin=π/2,ϕin = 0)
    Ue            = [us(x)+ui(x) for x in X].*isoutside
    solution      = vec([ [real(Ue[n])] for n in 1:length(Ue)])
    view_solution = gmsh.view.add("solution (real part)")
    gmsh.view.addModelData(view_solution,0,mname,"NodeData",output_tags,solution)
    gmsh.view.write(view_solution, "exact_solution.msh")
    return nothing
end

@info "computing exact solution..."
compute_exact_solution("xyplane.msh",k)
@info "done"

gmsh.finalize()
