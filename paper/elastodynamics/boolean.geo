SetFactory("OpenCASCADE");

// from http://en.wikipedia.org/wiki/Constructive_solid_geometry

Mesh.Algorithm = 6;
Mesh.CharacteristicLengthMin = 0.2;
Mesh.CharacteristicLengthMax = 0.2;

R = 1.4;
Rs = R*.7;
Rt = R*1.25;

Box(1) = {-R,-R,-R, 2*R,2*R,2*R};

Sphere(2) = {0,0,0,Rt};

BooleanIntersection(3) = { Volume{1}; Delete; }{ Volume{2}; Delete; };

Cylinder(4) = {-2*R,0,0, 4*R,0,0, Rs};
Cylinder(5) = {0,-2*R,0, 0,4*R,0, Rs};
Cylinder(6) = {0,0,-2*R, 0,0,4*R, Rs};

BooleanUnion(7) = { Volume{4}; Delete; }{ Volume{5,6}; Delete; };
BooleanDifference(8) = { Volume{3}; Delete; }{ Volume{7}; Delete; };