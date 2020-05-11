function single_double_layer(pde,X,Y=X;compress=Matrix,correction=:greenscorrection)
    Sop  = SingleLayerOperator(pde,X,Y)
    Dop  = DoubleLayerOperator(pde,X,Y)
    # convert to a possibly more efficient compress
    S = compress(Sop)
    D = compress(Dop)
    if correction == :greenscorrection
        # compute corrections
        δS = GreensCorrection(Sop,S,D)
        δD = GreensCorrection(Dop,δS.R,δS.L,δS.idxel_near) # reuse precomputed quantities of δS 
        return S+δS, D+δD
    elseif correction == :nothing
        return S,D
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
        return AD+δAD, H+δH
    elseif correction == :nothing
        return AD,H
    end
end

adjointdoublelayer(args...;kwargs...) = adjointdoublelayer_hypersingular(args...;kwargs...)[1]
hypersingular(args...;kwargs...)      = adjointdoublelayer_hypersingular(args...;kwargs...)[2]
