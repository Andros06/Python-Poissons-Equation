import scipy.sparse as sp
import scipy.sparse.linalg as lin
import numpy as np
import matplotlib.pyplot as plt

#antall punkter
m = 4
N = m + 2

#start og slutt punkt (Point a)
Pa = -1 
Pb = 1

#start og slutt punkt verdier (Point a Value)
PaV = 1
PbV = 1

x = np.linspace(Pa, Pb, N)

# avstand mellom punktene
h = x[1] - x[0]

L2 = (1/h**2) * sp.diags([1, -2, 1], [-1, 0, 1], shape=(N,N))

L2 = sp.csr_matrix(L2)


#Venstre randbetingelser
L2[0, 0] = -1/h
L2[0, 1] = -1/h

#Høgre randbetingelser
L2[-1, -1] = -1/h
L2[-1, -2] = -1/h

omega = 1

B = L2 + (omega**2) * sp.eye(N)

G = np.cos(np.pi * x)
# om vi hadde en funksjon f(x) på høyre side av ligningen ville vi satt G = f(x) istedet

# setter inn randbetingelsen i vektoren G
G[0]  +=  PaV
G[-1] -=  PbV

B_tett = B.toarray()

# løser systemet
V = lin.spsolve(B_tett, G)

plt.plot(x, V)
plt.xlabel('x')
plt.ylabel('U')
plt.title('Solution to Poisson Equation')
plt.show()

