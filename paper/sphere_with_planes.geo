SetFactory("OpenCASCADE");
Sphere(1) = {0, 0, 0, 1, -Pi/2, Pi/2, 2*Pi};
//+
Physical Surface("Sphere") = {1};
//+
Rectangle(2) = {-3, -3, 0, 6, 6, 0};
//+
Physical Surface("Output") = {2};
//+
Transfinite Surface {2} = {6, 5, 4, 3};

//+
Transfinite Curve {6, 5, 4, 7} = 40 Using Progression 1;
//+
Recombine Surface {2};
//+
Characteristic Length {2, 1} = 0.1;

