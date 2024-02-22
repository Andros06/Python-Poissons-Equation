import scipy.sparse as sp
import scipy.sparse.linalg as lin
import numpy as np

# antall punkter på innside
m = 4

# lager rutenettet, med to ekstra punkter ved randene
x = np.linspace(0,1,m+2)
# avstand mellom punktene
h = x[1] - x[0]

# Her er nøkkelen: scipy.sparse.diags(...) bygger en matrise med -2 på hoveddiagonal, og 1 på tilstøtende diagonalene.
# Det er lurt å bruke sparse matriser - vi ønsker ikke å lagre masse 0-er i minne
# Det blir særlig viktig i flere dimensjoner, hvor antall punkter blir veldig stort
# Du vil trolig komme deg unna i prosjektet med vanlige matriser, men det er uansett lettere å sette opp matrisen
# på denne måte

L = (1/h**2)*sp.diags([1,-2,1],[-1,0,1],shape=(m+2,m+2))

print(L.toarray())


omega = 1
a = 0
b = 2

# vi plusser L med omega^2 ganger identitetsmatrisen
A = L + (omega**2) * sp.eye(m+2)

F = np.cos(np.pi*x)

# hvis du vil ha f(x) ikke lik null skal den erstattes med
# F = f(x) for din funksjon f
# f eks, om du vil ha sin(x) skriver du F = np.sin(x)

F[0] = F[0] - a/(h**2)
F[-1] = F[-1] -b/(h**2)

# vi løser med sparse solver
U = lin.spsolve(A,F)

print(U)