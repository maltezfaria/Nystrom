using Nystrom

function compute_error(op,Γ)
    
end

function fig_gen()
    dim = 2
    niter = 5
    fig = plot()
    for qorder = (3,4,5)
        order = qorder+1.0
        OPERATORS = (Laplace(),Helmholtz(1))
        for  op in OPERATORS
            Γ = Nystrom.kite()
            Nystrom.refine!(Γ)
            nn, ee = compute_error(op,Γ,qorder,niter)
            op isa Laplace ? (opname = "Laplace") : (opname = "Helmholtz")
            plot!(fig,nn,ee,marker=:x,label="M=$qorder  "*opname)
            if op==last(OPERATORS)
                plot!(fig,nn,nn.^(-order)/nn[end]^(-order)*ee[end],xscale=:log10,yscale=:log10,
                      label="order $(Int(order)) slope",xlabel=L"$N_p$",ylabel=L"|error|_\infty",linewidth=4)
            end
        end
    end
    return fig
end

fig = fig_gen()
fname = "/home/lfaria/Dropbox/Luiz-Carlos/general_regularization/figures/fig1a.svg"
savefig(fig,fname)
