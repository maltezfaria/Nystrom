using Nystrom, FastGaussQuadrature, Plots, LinearAlgebra, IterativeSolvers, ParametricSurfaces
using ParametricSurfaces: getnodes
using Nystrom: error_interior_green_identity, error_exterior_green_identity, Point
using MAT

function near_field_error(pde,dim,p,h)
    geo = Acorn(rotation=(pi/2,0,0))
    fig = surface(clims=(-6,0),axis=false,colormap=:inferno,
               xlabel="x",ylabel="y",zlabel="z",legend=true)
    plot!(fig,geo)
    # construct exterior solution
    xin  = Point(0.0,0.1,0.)
    c    = pde isa Union{Helmholtz,Laplace} ? 1 : ones(dim)
    u    = (x)   -> SingleLayerKernel(pde)(xin,x)*c
    dudn = (x,n) -> transpose(DoubleLayerKernel(pde)(xin,x,n))*c
    # upper and bottom plane
    δ   = 0.1
    y   = -4:δ:4
    xup = 0:δ:4
    Xup = [(x,y,0.) for y in y, x in xup]
    Uup = [u(x) for x in Xup]
    xdown   = -4:δ:0
    Xdown   = [(x,y,0.) for y in y, x in xdown]
    Udown = [u(x) for x in Xdown]
    meshgen!(geo,h)
    Γ = quadgen(geo,p,gausslegendre)
    S,D = single_double_layer(pde,Γ)
    γ₀u       = γ₀(u,Γ)
    γ₁u       = γ₁(dudn,Γ)
    rhs       = S*γ₁u
    η     = im*pde.k
    σ,ch  = gmres(I/2 + D - η*S,γ₀u,verbose=false,log=true,tol=1e-14,restart=100)
    # error no correction
    SL, DL = single_double_layer(pde,Xup,Γ;correction=:nothing)
    U  = DL*σ - im*pde.k*SL*σ
    U  = reshape(U,length(y),length(xup))
    field_error_up = log10.(abs.(U-Uup))
    surface!(fig,xup,y,zero(field_error_up),surfacecolor=field_error_up,
             legend=false, clims=(-6,0))
    # heatmap!(fig,xup,y,field_error_up)
    # error correction
    SL, DL = single_double_layer(pde,Xdown,Γ)
    U  = DL*σ - im*pde.k*SL*σ
    U  = reshape(U,length(y),length(xdown))
    field_error_down = log10.(abs.(U-Udown))
    surface!(fig,xdown,y,zero(field_error_down),surfacecolor=field_error_down,
             legend=false,clims=(-6,0),colorscale=false)
    return fig
end

# figure 3d
#
# plotly()
hdf5()
dim       = 3
p         = 4
h         = 0.25
pde        = Helmholtz(dim=dim,k=2π)
fig       = near_field_error(pde,dim,p,h)
Plots.hdf5plot_write(fig, "plotsave.hdf5")
display(fig)
fname     = "/home/lfaria/Dropbox/Luiz-Carlos/general_regularization/draft/figures/nearfield_acorn.svg"
savefig(fig,fname)
