######################## 2D ##############################################
function ellipsis(;paxis=ones(2),center=zeros(2))
    f(s)       = center .+ paxis.*[cospi(s[1]),sinpi(s[1])]
    domain     = Cuboid(-1.0,1.0)
    surf       = ParametricSurface{2}(f,domain,[domain])
    return ParametricBody{2}([surf])
end
circle(;radius=1,center=zeros(2)) = ellipsis(;paxis=radius*ones(2),center=center)

function kite(;radius=1,center=zeros(2))
    f(s) = center .+ radius.*[cospi(s[1]) + 0.65*cospi(2*s[1]) - 0.65,
                1.5*sinpi(s[1])]
    domain = Cuboid(-1.0,1.0)
    surf   = ParametricSurface{2}(f,domain,[domain])
    return ParametricBody{2}([surf])
end

######################## 3D ##############################################
function cube(;paxis=ones(3),center=zeros(3))
    nparts = 6
    domain = Cuboid((-1.0,-1.0),(1.0,1.0))
    parts = ParametricSurface{2,3,Float64}[]
    for id=1:nparts
        param(x) = _cube_parametrization(x[1],x[2],id,paxis,center)
        patch = ParametricSurface{3}(param,domain,[domain])
        push!(parts,patch)
    end
    return ParametricBody{3}(parts)
end

function ellipsoid(;paxis=ones(3),center=zeros(3))
    nparts = 6
    domain = Cuboid((-1.0,-1.0),(1.0,1.0))
    parts = ParametricSurface{2,3,Float64}[]
    for id=1:nparts
        param(x) = _ellipsoid_parametrization(x[1],x[2],id,paxis,center)
        patch = ParametricSurface{3}(param,domain,[domain])
        push!(parts,patch)
    end
    return ParametricBody{3}(parts)
end
sphere(;radius=1,center=zeros(3)) = ellipsoid(;paxis=radius*ones(3),center=center)

function bean(;paxis=ones(3),center=zeros(3))
    nparts = 6
    domain = Cuboid((-1.0,-1.0),(1.0,1.0))
    parts  = ParametricSurface{2,3,Float64}[]
    for id=1:nparts
        param(x) = _bean_parametrization(x[1],x[2],id,paxis,center)
        patch    = ParametricSurface{3}(param,domain,[domain])
        push!(parts,patch)
    end
    return ParametricBody{3}(parts)
end

function _cube_parametrization(u,v,id,paxis,center)
    if id==1
        x = [1.,u,v]
    elseif id==2
        x = [-u,1.,v];
    elseif id==3
        x = [u,v,1.];
    elseif id==4
        x =[-1.,-u,v];
    elseif id==5
        x = [u,-1.,v];
    elseif id==6
        x = [-u,v,-1.];
    end
    return center .+ paxis.*x
end


function _sphere_parametrization(u,v,id,rad=1,center=zeros(3))
    if id==1
        x = [1.,u,v]
    elseif id==2
        x = [-u,1.,v];
    elseif id==3
        x = [u,v,1.];
    elseif id==4
        x =[-1.,-u,v];
    elseif id==5
        x = [u,-1.,v];
    elseif id==6
        x = [-u,v,-1.];
    end
    return center .+ rad.*x./sqrt(u^2+v^2+1)
end

function _ellipsoid_parametrization(u,v,id,paxis=ones(3),center=zeros(3))
    x = _sphere_parametrization(u,v,id)
    return x .* paxis .+ center
end

function _bean_parametrization(u,v,id,paxis=one(3),center=zeros(3))
    x = _sphere_parametrization(u,v,id)
    a = 0.8; b = 0.8; alpha1 = 0.3; alpha2 = 0.4; alpha3=0.1
    x[1] = a*sqrt(1.0-alpha3*cospi(x[3])).*x[1]
    x[2] =-alpha1*cospi(x[3])+b*sqrt(1.0-alpha2*cospi(x[3])).*x[2];
    x[3] = x[3];
    return x .* paxis .+ center
end
