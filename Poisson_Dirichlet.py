import scipy.sparse as sp
import scipy.sparse.linalg as lin
import numpy as np
import matplotlib.pyplot as plt

# Antall punkter, N inkluderer star og slutt punkt
m = 5
N = m + 2

# Start og slutt punkt (Fra punkt a til punkt b)
Pa = -1 
Pb = 1

# Start og slutt punkt verdier (Verdi for punkt a og b)
PaV = 0
PbV = 2

# Lager x verdier mellom Pa og Pb med N antall punkt
x = np.linspace(Pa, Pb, N)

# Finner avstand mellom punktene
h = x[1] - x[0]

# Lager sparsmatrise med andrederivert ved "Finitedifference-metode"
L = (1/h**2) * sp.diags([1, -2, 1], [-1, 0, 1], shape=(m,m))

# Konverterar til vanlig matrise
A = L.toarray()

# Lager til eit array med berre x verdien vi vil finne
x_intern = x[1:-1]

# Funksjon fra oppgåven
F = np.cos(np.pi * x_intern)

# Legger til Dirichlet randbetingelser i F
F[0] = F[0] - (PaV / (h**2))
F[-1]= F[-1] - (PbV / (h**2))

# Løyser for A og F
u = np.linalg.solve(A, F)

# Legger til Dirichlet randbetingelser
u_full = np.hstack([PaV, u, PbV])

# Plotting for graf
plt.plot(x, u_full)
plt.xlabel("x")
plt.ylabel("u(x)")
plt.title("Dirichlet Løysning")
plt.grid(True)
plt.show()
