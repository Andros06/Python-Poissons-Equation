import scipy.sparse as sp
import scipy.sparse.linalg as lin
import numpy as np
import matplotlib.pyplot as plt

# antall punkter
m = 5
N = m + 2

#s tart og slutt punkt
Pa = -1 
Pb = 1

# Setter verdier i start og slutt punkt
PaV = 0
PbV = 2

# Danner X
x = np.linspace(Pa, Pb, N)

# Avstand mellom punktene
h = x[1] - x[0]

L = (1/h**2) * sp.diags([1, -2, 1], [-1, 0, 1], shape=(m,m))

A = L.toarray()

x_intern = x[1:-1]

F = np.cos(np.pi * x_intern)


F[0] = F[0] - (PaV / (h**2))
F[-1]= F[-1] - (PbV / (h**2))

u = np.linalg.solve(A, F)

u_full = np.hstack([PaV, u, PbV])

print(u_full)

plt.plot(x, u_full)
plt.xlabel("x")
plt.ylabel("u(x)")
plt.title("Dirichlet Løysning")
plt.grid(True)
plt.show()
