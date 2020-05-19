using Nystrom, Plots, GmshTools

function fig_gen()
    dim = 3
    R = 1
    qorder    = (1,2)
    h0        = 0.25

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.option.setNumber("Mesh.Algorithm",8)
    fig       = plot(yscale=:log10,xscale=:log10,xlabel="√N",ylabel="error")
    niter     = 3
    for p in qorder
        gmsh.clear()
        geo = gmsh.model.occ.addSphere(0,0,0,1)
        gmsh.model.occ.synchronize()
        Nystrom.gmsh_set_meshsize(h0)
        gmsh.model.mesh.generate(2)
        gmsh.model.mesh.recombine()
        conv_order = p + 1
        dof = []
        ee = []
        for _ in 1:niter
            gmsh.model.mesh.setOrder(conv_order+1) # set it higher than the method's error to avoid geometrical errors
            Γ     = quadgengmsh("Gauss$p",2,-1)
            push!(dof,length(Γ))
            push!(ee,abs(sum(Γ.weights) - 4π))
            gmsh.model.mesh.refine()
        end
        dof = sqrt.(dof) # dof per dimension, roughly inverse of mesh size
        plot!(fig,dof,ee,label="p=$p",m=:o,color=p)
        plot!(fig,dof,1 ./(dof.^conv_order)*dof[end]^(conv_order)*ee[end],
              label="",linewidth=4,color=p)
    end
    gmsh.finalize()
    return fig
end

fig = fig_gen()
# fname = "/tmp/gmsh_test.pdf"
# savefig(fig,fname)
display(fig)
