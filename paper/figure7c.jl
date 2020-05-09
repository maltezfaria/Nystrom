using Nystrom, FastGaussQuadrature, Plots, LinearAlgebra, IterativeSolvers
using Nystrom: Point, getnodes

function fig_gen()
    dim = 2
    Γ     = Domain(dim=dim)
    geo   = kite()
    push!(Γ,geo)
    xtest = [Point(x,y) for x in -5:0.1:5, y in -5:0.1:5]
    k         = 2π
    p         = 3
    h0        = 0.1
    niter     = 1
    pde       = Helmholtz(dim=dim,k=k)
    x = y = -2:0.025:2
    X = [Point(x,y) for y in y, x in x]
    # construct exact exterior solution
    xin  = (0.1,-0.1)
    u    = (x)   -> SingleLayerKernel(pde)(xin,x)
    dudn = (x,n) -> DoubleLayerKernel(pde)(xin,x,n)
    figs = []
    conv_order = p + 1
    meshgen!(Γ,h0)
    quadgen!(Γ,p;algo1d=gausslegendre)
    S,D   = single_double_layer(pde,Γ)
    γ₀u   = γ₀(u,Γ)
    γ₁u   = γ₁(dudn,Γ)
    σ,ch  = gmres(I/2 + D - im*k*S,γ₀u,verbose=false,log=true,tol=1e-14,restart=100)
    # compute the trace error on γ₁u
    ADL,H = adjointdoublelayer_hypersingular(pde,Γ)
    # compute error in the field
    poly = getnodes(Γ) |> deepcopy
    push!(poly,poly[1])
    isoutside = [in(pt,poly) ? -Inf : 1 for pt in X]
    SL, DL = single_double_layer(pde,X,Γ;correction=:greenscorrection)
    U  = DL*σ - im*k*SL*σ
    U  = reshape(U,length(y),length(x))
    Ue = [u((x,y)) for y in y, x in x]
    field_error = log.(abs.(U-Ue)) .* isoutside
    fig    = heatmap(x,y,field_error,
                     xlabel="x",ylabel="y",
                     clims=(-16.,-2.),
                     zscale=:log10,
                     size=(450,400))
    plot!(fig,Γ,lc=:black,lw=2)
    @show length(Γ), norm(γ₁u - (H*σ - im*k*(ADL*σ - σ/2)),Inf)
    return fig
end

fig = fig_gen()
fname = "/home/lfaria/Dropbox/Luiz-Carlos/general_regularization/draft/figures/fig7c.pdf"
savefig(fig,fname)
display(fig)
