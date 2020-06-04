using JLD, LaTeXStrings, Plots

conv_order = 3
fig = plot(yscale=:log10,xscale=:log10,xlabel= L"\sqrt{N}",ylabel="error",legend=:topright,
           framestyle=:box,xtickfontsize=10,ytickfontsize=10);

dict = load("convergence_torus.jld")
h = dict["h"]
dof = dict["dof"]
eFar = dict["eFar"]
dof_per_dim = sqrt.(dof)
# plot!(fig,h[2:end],eFar, m=:x,label="sphere")
plot!(fig,dof_per_dim,eFar, m=:x,label="torus")

dict = load("convergence_sphere.jld")
h = dict["h"]
dof = dict["dof"]
eFar = dict["eFar"]
dof_per_dim = sqrt.(dof)
# plot!(fig,h[2:end],eFar, m=:x,label="sphere")
plot!(fig,dof_per_dim,eFar, m=:x,label="sphere")

plot!(fig,dof_per_dim,1 ./(dof_per_dim.^conv_order)*dof_per_dim[2]^(conv_order)*eFar[2],
      label="third order slope",linewidth=4,line=:dot)

display(fig)

fname = "/home/lfaria/Dropbox/Luiz-Carlos/general_regularization/draft/figures/elastodynamic_convergence_scattering.pdf"
savefig(fig,fname)
