import scipy.sparse as sp
import scipy.sparse.linalg as lin
import numpy as np
import matplotlib.pyplot as plt

# Antall punkter, N inkluderer star og slutt punkt
m = 4
N = m + 2

# Start og slutt punkt (Fra punkt a til punkt b)
Pa = -1 
Pb = 1

# Start og slutt punkt verdier (Verdi for punkt a og b)
PaV = 1
PbV = 1

# Lager x verdier mellom Pa og Pb med N antall punkt
x = np.linspace(Pa, Pb, N)

# Finner avstand mellom punktene
h = x[1] - x[0]

# Lager til den andrederiverte matrisen L2 
# Lager først sparsmatrise med andrederivert ved "Finitedifference-metode"
L2 = (1/h**2) * sp.diags([1, -2, 1], [-1, 0, 1], shape=(N,N))

# Konvertere til komprimert matrise for effektivitet
L2 = sp.csr_matrix(L2)


# Venstre randbetingelser
L2[0, 0] = -1/h
L2[0, 1] =  1/h

# Høgre randbetingelser
L2[-1, -1] = 1/h
L2[-1, -2] = -1/h

# Omega = 0 pga poisson likning
omega = 0

B = L2 + (omega**2) * sp.eye(N)

# Funksjonen fra oppgåve
G = np.cos(np.pi * x)

# setter inn randbetingelsen i vektoren G
G[0]  =  PaV
G[-1] =  PbV

# Løyser likningen 
V, istop, itn, r1norm = lin.lsqr(L2, G)[:4]

# Plotting for graf
plt.plot(x, V)
plt.xlabel('x')
plt.ylabel('u(x)')
plt.title('Neumann løysning')
plt.grid(True)
plt.show()

