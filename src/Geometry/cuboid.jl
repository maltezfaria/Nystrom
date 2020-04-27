"""
    Cuboid{N,T}

Hyperrectangle in `N` dimensions given by `low_corner::Point{N,T}` and `high_corner::Point{N,T}`
"""
struct Cuboid{N,T}
    low_corner::Point{N,T}
    high_corner::Point{N,T}
end

Cuboid(a,b) = Cuboid(Point(a),Point(b))

Base.:(==)(h1::Cuboid, h2::Cuboid) = (h1.low_corner == h2.low_corner) && (h1.high_corner == h2.high_corner)
Base.in(point,h::Cuboid)                   = all(h.high_corner .>= point .>= h.low_corner)
Base.eltype(h::Cuboid{N,T}) where {N,T}    = T
dimension(h::Cuboid{N}) where {N}          = N

################################################################################
## CONVENIENCE FUNCTIONS FOR CUBOIDS
################################################################################
"""
    getpoints(Cuboid{N,T})

Return all `2^N` points composing the cuboid.
"""
function getpoints(c::Cuboid{N}) where {N}
    l = c.low_corner
    h = c.high_corner
    if N==1
        return l,h
    elseif N==2
        x1,y1 = l[1],l[2]
        x2,y2 = h[1],h[2]
        return Point(x1,y1),
        Point(x1,y2),
        Point(x2,y1),
        Point(x2,y2)
    end
end
# TODO: write the get points above in a generic way

"""
    split(rec::Cuboid,axis::Int,place)

Split a hyperrectangle in two along the `axis` direction at the  position `place`. Returns the two resulting hyper-rectangles.
"""
function split(rec::Cuboid,axis,place)
    N            = dimension(rec)
    T            = eltype(rec)
    high_corner1 = ntuple(n-> n==axis ? place : rec.high_corner[n], N)
    low_corner2  = ntuple(n-> n==axis ? place : rec.low_corner[n], N)
    rec1         = Cuboid{N,T}(rec.low_corner, high_corner1)
    rec2         = Cuboid{N,T}(low_corner2,rec.high_corner)
    return (rec1, rec2)
end

"""
    split(rec::Cuboid,axis)
When no `place` is given, defaults to splitting in the middle of the axis.
"""
function split(rec::Cuboid,axis)
    place              = (rec.high_corner[axis] + rec.low_corner[axis])/2
    split(rec,axis,place)
end

"""
    split(rec::Cuboid)
When no axis and no place is given, defaults to splitting along the largest axis
"""
function split(rec::Cuboid)
    axis = argmax(rec.high_corner .- rec.low_corner)
    split(rec,axis)
end

diameter(cub::Cuboid) = norm(cub.high_corner - cub.low_corner,2)

function distance(rec1::Cuboid{N},rec2::Cuboid{N}) where {N}
    d2 = 0
    for i=1:N
        d2 += max(0,rec1.low_corner[i] - rec2.high_corner[i])^2 + max(0,rec2.low_corner[i] - rec1.high_corner[i])^2
    end
    return sqrt(d2)
end

function bounding_box(data)
    pt_min = minimum(data)
    pt_max = maximum(data)
    return Cuboid(pt_min,pt_max)
end
container(data) = bounding_box(data)

center(rec::Cuboid) = (rec.low_corner + rec.high_corner) ./ 2
radius(rec::Cuboid) = diameter(rec) ./ 2

################################################################################
## PLOTTING RECIPES
################################################################################
@recipe function f(rec::Cuboid{N}) where {N}
    seriestype := :path
    label := ""
    if N == 1
        pt1 = rec.low_corner
        pt2 = rec.high_corner
        x1, x2 = pt1[1],pt2[1]
        [x1,x2], [0, 0.]
    elseif N == 2
        pt1 = rec.low_corner
        pt2 = rec.high_corner
        x1, x2 = pt1[1],pt2[1]
        y1, y2 = pt1[2],pt2[2]
        [x1,x2,x2,x1,x1],[y1,y1,y2,y2,y1]
    elseif N == 3
        seriestype := :path
        pt1 = rec.low_corner
        pt2 = rec.high_corner
        x1, x2 = pt1[1],pt2[1]
        y1, y2 = pt1[2],pt2[2]
        z1, z2 = pt1[3],pt2[3]
        @series [x1,x2,x2,x1,x1],[y1,y1,y2,y2,y1],[z1,z1,z1,z1,z1]
        @series [x1,x2,x2,x1,x1],[y1,y1,y2,y2,y1],[z2,z2,z2,z2,z2]
        @series [x1,x1],[y1,y1],[z1,z2]
        @series [x2,x2],[y1,y1],[z1,z2]
        @series [x2,x2],[y2,y2],[z1,z2]
        @series [x1,x1],[y2,y2],[z1,z2]
    end
end
