using Nystrom, FastGaussQuadrature, Plots, LinearAlgebra, IterativeSolvers, ParametricSurfaces
using ParametricSurfaces: getnodes
using Nystrom: error_interior_green_identity, error_exterior_green_identity, Point

function near_field_error(pde,dim,p,h)
    geo = dim == 2 ? Kite() : Bean()
    fig = plot(size=(450,400),clims=(-6,0),framestyle=:box)
    # construct exterior solution
    xin  = Point(0.0,0.)
    c    = pde isa Union{Helmholtz,Laplace} ? 1 : ones(dim)
    u    = (x)   -> SingleLayerKernel(pde)(xin,x)*c
    dudn = (x,n) -> transpose(DoubleLayerKernel(pde)(xin,x,n))*c
    # upper and bottom plane
    δ = 0.01
    x   = -2:δ:2
    yup   = 0:δ:2
    Xup = dim == 2 ? [(x,y) for y in yup, x in x] : [(x,y,0.) for y in yup, x in x]
    Uup = [u(x) for x in Xup]
    ydown   = -2:δ:0
    Xdown = dim == 2 ? [(x,y) for y in ydown, x in x] : [(x,y,0.) for y in ydown, x in x]
    Udown = [u(x) for x in Xdown]
    meshgen!(geo,h)
    Γ = quadgen(geo,p,gausslegendre)
    S,D = single_double_layer(pde,Γ)
    γ₀u       = γ₀(u,Γ)
    γ₁u       = γ₁(dudn,Γ)
    rhs       = S*γ₁u
    η     = im*pde.k
    σ,ch  = gmres(I/2 + D - η*S,γ₀u,verbose=false,log=true,tol=1e-14,restart=100)
    poly = getnodes(Γ) |> deepcopy
    push!(poly,poly[1])
    # error no correction
    isoutside = [in(pt,poly) ? 0 : 1 for pt in Xup]
    SL, DL = single_double_layer(pde,Xup,Γ;correction=:nothing)
    U  = DL*σ - im*pde.k*SL*σ
    U  = reshape(U,length(yup),length(x))
    field_error = log10.(abs.(U-Uup) .* isoutside)
    heatmap!(fig,x,yup,field_error)
    # error correction
    isoutside = [in(pt,poly) ? -Inf : 1 for pt in Xdown]
    SL, DL = single_double_layer(pde,Xdown,Γ)
    U  = DL*σ - im*pde.k*SL*σ
    U  = reshape(U,length(ydown),length(x))
    field_error = log10.(abs.(U-Udown)) .* isoutside
    heatmap!(fig,x,ydown,field_error)
    meshgen!(geo,1e-2)
    plot!(fig,geo,lc=:black,lw=2)
    return fig
end

# figure 3c
dim       = 2
p         = 4
h         = 0.1
pde       = Helmholtz(dim=dim,k=2π)
fig       = near_field_error(pde,dim,p,h)
fname     = "/home/lfaria/Dropbox/Luiz-Carlos/general_regularization/draft/figures/fig3c.pdf"
savefig(fig,fname)
display(fig)
