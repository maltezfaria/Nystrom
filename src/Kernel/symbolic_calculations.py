import sympy
from sympy import *

x1, x2, x3, y1, y2, y3 = symbols("x1 x2 x3 y1 y2 y3")

# Laplace2d (for testing)
d = sqrt((x1-y1)**2 + (x2-y2)**2)
G = -log(d)

df = vector.grad(G,coords=(x1,x2))

# print df

print("done")
