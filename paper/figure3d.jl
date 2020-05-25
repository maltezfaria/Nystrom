using Nystrom, FastGaussQuadrature, Plots, LinearAlgebra, IterativeSolvers, ParametricSurfaces
using ParametricSurfaces: getnodes
using Nystrom: error_interior_green_identity, error_exterior_green_identity

function near_field_error(pde,dim,p,h)
    geo = Bean(;paxis=(-1,-1,1))
    fig = plot(size=(800,800),clims=(-6,0),camera=(90,-80),axis=[],grid=false,legend=:none,colormap=:inferno,framestyle=:box)
    # construct exterior solution
    xin  = (0.0,0.1,0.)
    c    = pde isa Union{Helmholtz,Laplace} ? 1 : ones(dim)
    u    = (x)   -> SingleLayerKernel(pde)(xin,x)*c
    dudn = (x,n) -> transpose(DoubleLayerKernel(pde)(xin,x,n))*c
    # upper and bottom plane
    δ   = 0.05
    y   = -1.2:δ:1.2
    xup = 0:δ:1.2
    Xup = [(x,y,0.) for y in y, x in xup]
    Uup = [u(x) for x in Xup]
    xdown   = -1.2:δ:0
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
    field_error = log10.(abs.(U-Uup))
    surface!(fig,xup,y,zero(field_error),surfacecolor=field_error)
    # error correction
    SL, DL = single_double_layer(pde,Xdown,Γ)
    U  = DL*σ - im*pde.k*SL*σ
    U  = reshape(U,length(y),length(xdown))
    field_error = log10.(abs.(U-Udown))
    surface!(fig,xdown,y,zero(field_error),surfacecolor=field_error)
    return fig
end

# figure 3d
# pyplot()
plotlyjs()
dim       = 3
p         = 4
h         = 0.25
pde        = Helmholtz(dim=dim,k=2π)
fig       = near_field_error(pde,dim,p,h)

function plot_geo(fig,geo,h=0.1)
    for patch in geo.parts
        domain = patch.domain
        xmin,ymin = domain.origin
        xmax,ymax   = domain.origin + domain.widths
        xrange = xmin:h:xmax
        yrange = ymin:h:ymax
        pts    = [patch.parametrization((x,y)) for x in xrange, y in yrange]
        x      =  [pt[1] for pt in pts]
        y      =  [pt[2] for pt in pts]
        z      =  [pt[3] for pt in pts]
        surface!(fig,x,y,z,color=:gray50)
    end
    return fig
end

geo = Bean(;paxis=(-1,-1,1))
plot_geo(fig,geo)

display(fig)

fname     = "/home/lfaria/Dropbox/Luiz-Carlos/general_regularization/draft/figures/fig3d.svg"
savefig(fig,fname)
