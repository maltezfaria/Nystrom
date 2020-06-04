using gmsh, LinearAlgebra
using Nystrom: Point, sphere_helmholtz_soundsoft
const PATH = @__DIR__

gmsh.initialize()

gmsh.option.setNumber("General.Terminal",1)

function generate_A319_mesh(order,h,recombine)
    gmsh.clear()
    gmsh.model.add("A319")
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax",h)
    gmsh.merge(joinpath(PATH,"A319.brep"))
    # generate and write
    gmsh.model.mesh.generate(2)
    recombine && gmsh.model.mesh.recombine()
    gmsh.model.mesh.setOrder(order)
    fname = "A319_order_$(order)_recombine_$(recombine)_h_$(Int(h)).msh"
    gmsh.write(fname)
end

λ = 2e3
k = 2π/λ
h = λ/10

# generate a mesh of the xyplane for outputting the solution
gmsh.clear()
gmsh.model.add("output")
lx = -1e4; wx = 6e4
ly = -2e4; wy = 4e4
lz = -2e4; wz = 4e4
p1 = gmsh.model.occ.addPoint(lx,ly,lz)
p2 = gmsh.model.occ.addPoint(lx+wx,ly,lz)
p3 = gmsh.model.occ.addPoint(lx+wx,ly+wy,lz)
p4 = gmsh.model.occ.addPoint(lx,ly+wy,lz)
l1 = gmsh.model.occ.addLine(p1,p2)
l2 = gmsh.model.occ.addLine(p2,p3)
l3 = gmsh.model.occ.addLine(p3,p4)
l4 = gmsh.model.occ.addLine(p4,p1)
loop = gmsh.model.occ.addCurveLoop([l1,l2,l3,l4])
gmsh.model.occ.addPlaneSurface([loop])
#
p1 = gmsh.model.occ.addPoint(lx+wx,ly,lz)
p2 = gmsh.model.occ.addPoint(lx+wx,ly+wy,lz)
p3 = gmsh.model.occ.addPoint(lx+wx,ly+wy,lz+wz)
p4 = gmsh.model.occ.addPoint(lx+wx,ly,lz+wz)
l1 = gmsh.model.occ.addLine(p1,p2)
l2 = gmsh.model.occ.addLine(p2,p3)
l3 = gmsh.model.occ.addLine(p3,p4)
l4 = gmsh.model.occ.addLine(p4,p1)
loop = gmsh.model.occ.addCurveLoop([l1,l2,l3,l4])
gmsh.model.occ.addPlaneSurface([loop])
#
p1 = gmsh.model.occ.addPoint(lx,ly+wy,lz)
p2 = gmsh.model.occ.addPoint(lx+wx,ly+wy,lz)
p3 = gmsh.model.occ.addPoint(lx+wx,ly+wy,lz+wz)
p4 = gmsh.model.occ.addPoint(lx,ly+wy,lz+wz)
l1 = gmsh.model.occ.addLine(p1,p2)
l2 = gmsh.model.occ.addLine(p2,p3)
l3 = gmsh.model.occ.addLine(p3,p4)
l4 = gmsh.model.occ.addLine(p4,p1)
loop = gmsh.model.occ.addCurveLoop([l1,l2,l3,l4])
gmsh.model.occ.addPlaneSurface([loop])
gmsh.model.occ.synchronize()
gmsh.option.setNumber("Mesh.CharacteristicLengthMax",4e2)
gmsh.model.mesh.generate(2)
gmsh.model.mesh.recombine()
gmsh.write("output.msh")

hsurf = 2e2
# generate_A319_mesh(1,hsurf,false)
# generate_A319_mesh(1,hsurf,true)
# generate_A319_mesh(1,2*hsurf,false)
generate_A319_mesh(1,hsurf/2,false)
# generate_A319_mesh(1,2*hsurf,true)
# generate_A319_mesh(2,hsurf,false)
# generate_A319_mesh(2,hsurf,true)

gmsh.finalize()
