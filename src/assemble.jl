function single_double_layer(pde,X,Y=X;compress=Matrix,correction=:greenscorrection)
    Sop  = SingleLayerOperator(pde,X,Y)
    Dop  = DoubleLayerOperator(pde,X,Y)
    # convert to a possibly more efficient format
    S = compress(Sop)
    D = compress(Dop)
    if correction == :greenscorrection
        # compute corrections
        δS = GreensCorrection(Sop,S,D)
        δD = GreensCorrection(Dop,δS.R,δS.L,δS.idxel_near)  # reuse precomputed quantities of δS
        δS_sparse = sparse(δS)
        δD_sparse = sparse(δD)
        # add corrections to the dense part
        axpy!(true,δS_sparse,S)
        axpy!(true,δD_sparse,D)
        return S,D
    elseif correction == :nothing
        return S,D
    else
        error("unrecognized correction method")
    end
end
singlelayer(args...;kwargs...) = single_double_layer(args...;kwargs...)[1]
doublelayer(args...;kwargs...) = single_double_layer(args...;kwargs...)[2]

function adjointdoublelayer_hypersingular(pde,X,Y=X;compress=Matrix,correction=:greenscorrection)
    ADop  = AdjointDoubleLayerOperator(pde,X,Y)
    Hop   = HyperSingularOperator(pde,X,Y)
    # convert to a possibly more efficient compress
    AD = compress(ADop)
    H  = compress(Hop)
    if correction == :greenscorrection
        # compute corrections
        δAD = GreensCorrection(ADop,AD,H)
        δH  = GreensCorrection(Hop,δAD.R,δAD.L,δAD.idxel_near) # reuse precomputed quantities of δAD
        δAD_sparse = sparse(δAD)
        δH_sparse = sparse(δH)
        axpy!(true,δAD_sparse,AD)
        axpy!(true,δH_sparse,H)
        return AD, H
    elseif correction == :nothing
        return AD,H
    else
        error("unrecognized correction method")
    end
end

adjointdoublelayer(args...;kwargs...) = adjointdoublelayer_hypersingular(args...;kwargs...)[1]
hypersingular(args...;kwargs...)      = adjointdoublelayer_hypersingular(args...;kwargs...)[2]

function _single_double_layer(pde,X,Y=X;compress=Matrix,correction=:greenscorrection,xs)
    Sop  = SingleLayerOperator(pde,X,Y)
    Dop  = DoubleLayerOperator(pde,X,Y)
    # convert to a possibly more efficient format
    S = compress(Sop)
    D = compress(Dop)
    if correction == :greenscorrection
        # compute corrections
        δS = GreensCorrection(Sop,S,D,xs)
        δD = GreensCorrection(Dop,δS.R,δS.L,δS.idxel_near)  # reuse precomputed quantities of δS
        δS_sparse = sparse(δS)
        δD_sparse = sparse(δD)
        # add corrections to the dense part
        axpy!(true,δS_sparse,S)
        axpy!(true,δD_sparse,D)
        return S,D
    elseif correction == :nothing
        return S,D
    else
        error("unrecognized correction method")
    end
end
