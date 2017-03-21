import math
import numpy as np
from scipy import sparse, linalg

class Implicit_EuPut(object):
    def __init__(self, int_rate: object, vol: object, time: object, Strike: object, SMax: object, Time_Steps: object, Price_Steps: object, Spot: object) -> object:
        self.r = int_rate
        self.v = vol
        self.T = time
        self.K = Strike
        self.S = SMax
        self.Nt = Time_Steps
        self.Ns = Price_Steps
        self.S0 = Spot
        self.dt = float(self.T) / float(self.Nt)
        self.ds = float(self.S) / float(self.Ns)

    def implicit_finite(self):
        #construct the time T payoff series
        zero = np.array([0.0 for i in range(self.Ns - 1)])
        spot = np.array([(i + 1) * self.ds for i in range(self.Ns - 1)])
        strike = np.array([float(self.K) for i in range(self.Ns - 1)])
        V = np.array(zero <= (strike-spot), dtype = int)
        #V = np.maximum(zero, strike-spot)

        V_0 = np.array([math.exp(-self.r*self.dt*n) for n in range(self.Nt+1)])

        #construct the matrix
        ai = np.array([(self.v**2*i**2/4 - self.r*i/4) for i in range(self.Ns+1)])
        bi = np.array([(self.v**2*i**2/2 + self.r/2) for i in range(self.Ns+1)])
        ci = np.array([(self.v**2*i**2/4 + self.r*i/4) for i in range(self.Ns+1)])
        sdt = np.array([-1.0/self.dt for i in range(self.Ns-1)])
        #A = sparse.spdiags([ai[2:], sdt[:]-bi[1:self.Ns], ci[:self.Ns-1]], [-1,0,1], self.Ns-1, self.Ns-1).toarray()
        B = sparse.spdiags([-ai[2:], sdt[:]+bi[1:self.Ns], -ci[:self.Ns-1]], [-1,0,1], self.Ns-1, self.Ns-1).toarray()
        AB = np.asarray([ci[:self.Ns-1],sdt[:]-bi[1:self.Ns],np.insert(ai,self.Ns,0)[2:self.Ns+1]])

        #print(AB)

        F0 = -ai[1]*(V_0[:self.Nt]+V_0[1:])

        for i in range(self.Nt):
            w = B.dot(V)
            w[0] = w[0] + F0[i]
            V = linalg.solve_banded((1,1),AB, w)

        return V[self.S0/self.ds-1]

a=Implicit_EuPut(0.03, 0.2, 1, 100, 300, 1200, 1200, 100)
print(a.implicit_finite())

#scipy.linalg.solve_banded(l_and_u, ab, b, overwrite_ab=False, overwrite_b=False, debug=False, check_finite=True)[source]








