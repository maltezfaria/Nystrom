using gmsh, LinearAlgebra
path = @__DIR__
gmsh.initialize()
gmsh.clear()
λ = 0.5e3
h = λ/10 # approximate element size for plotting the field
k = 2π/λ
order = 1
recombine = false

gmsh.option.setNumber("General.Terminal", 1)
gmsh.option.setNumber("Mesh.CharacteristicLengthMax",h)
# generate a mesh of the xyplane for outputting the solution
gmsh.model.add("xyplane")
lx = -2e4; wx = 8e4
ly = -2e4; wy = 4e4
rec = gmsh.model.occ.addRectangle(lx,ly,0,wx,wy)
gmsh.model.occ.synchronize()
gmsh.model.mesh.setTransfiniteSurface(rec)
gmsh.model.mesh.setRecombine(2,rec)
Nx,Ny = ceil(Int,wx/h), ceil(Int,wy/h)
gmsh.model.mesh.setTransfiniteCurve(1,Nx)
gmsh.model.mesh.setTransfiniteCurve(2,Ny)
gmsh.model.mesh.setTransfiniteCurve(3,Nx)
gmsh.model.mesh.setTransfiniteCurve(4,Ny)
gmsh.model.mesh.generate(2)
gmsh.write("xyplane.msh")

gmsh.clear()
gmsh.model.add("xzplane")
lx = -2e4; wx = 8e4
ly = -2e4; wy = 4e4
rec = gmsh.model.occ.addRectangle(lx,ly,0,wx,wy)
gmsh.model.occ.rotate([(2,rec)],0,1,0,1,0,0,π/2)
gmsh.model.occ.synchronize()
gmsh.model.mesh.setTransfiniteSurface(rec)
gmsh.model.mesh.setRecombine(2,rec)
Nx,Ny = ceil(Int,wx/h), ceil(Int,wy/h)
gmsh.model.mesh.setTransfiniteCurve(1,Nx)
gmsh.model.mesh.setTransfiniteCurve(2,Ny)
gmsh.model.mesh.setTransfiniteCurve(3,Nx)
gmsh.model.mesh.setTransfiniteCurve(4,Ny)
gmsh.model.mesh.generate(2)
gmsh.write("xzplane.msh")

# h = λ/10 # approximate element size
h   = 100
gmsh.option.setNumber("Mesh.CharacteristicLengthMax",h)

gmsh.clear()
gmsh.model.add("A319")
gmsh.merge(joinpath(path,"A319.brep"))
gmsh.model.mesh.generate(2)
recombine && gmsh.model.mesh.recombine()
gmsh.model.mesh.setOrder(order)
gmsh.write("A319_order_$(order)_recombine_$(recombine).msh")

# h = h/2 # refined element size
# gmsh.option.setNumber("Mesh.CharacteristicLengthMax",h)
# gmsh.clear()
# gmsh.model.add("A319_refined")
# gmsh.merge(joinpath(path,"A319.brep"))
# gmsh.model.mesh.generate(2)
# recombine && gmsh.model.mesh.recombine()
# gmsh.model.mesh.setOrder(order)
# gmsh.write("A319_refined.msh")

gmsh.finalize()
