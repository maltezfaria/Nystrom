function single_double_layer(pde,X,Y=X;format=Matrix)
    Sop  = SingleLayerOperator(pde,X,Y)
    Dop  = DoubleLayerOperator(pde,X,Y)
    # convert to a possibly more efficient format
    S = format(Sop)
    D = format(Dop)
    # compute corrections
    δS = GreensCorrection(Sop,S,D)
    δD = GreensCorrection(Dop,δS.R,δS.L,δS.idxel_near)
    return S+δS, D+δD
end

singlelayer(args...;kwargs...) = single_double_layer(args...;kwargs...)[1]
doublelayer(args...;kwargs...) = single_double_layer(args...;kwargs...)[2]

function adjointdoublelayer_hypersingular(pde,X,Y=X;format=Matrix)
    ADop  = AdjointDoubleLayerOperator(pde,X,Y)
    Hop   = HyperSingularOperator(pde,X,Y)
    # convert to a possibly more efficient format
    AD = format(ADop)
    H  = format(Hop)
    # compute corrections
    δAD = GreensCorrection(ADop,AD,H)
    δH  = GreensCorrection(Hop,δAD.R,δAD.L,δAD.idxel_near)
    return AD+δAD, H+δH
end

adjointdoublelayer(args...;kwargs...) = adjointdoublelayer_hypersingular(args...;kwargs...)[1]
hypersingular(args...;kwargs...)      = adjointdoublelayer_hypersingular(args...;kwargs...)[2]
