# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 14:15:32 2017

@author: e0046450
"""

import math
import numpy as np
from scipy import sparse, linalg
import matplotlib.pyplot as plt

class Implicit_EuDigi(object):
    def __init__(self, opt_type, int_rate, vol, time, Strike, SMax, Time_Steps, Price_Steps, Spot):
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
        self.type = opt_type

    def implicit_finite_put(self):
        #construct the time T payoff series
        zero = np.array([0.0 for i in range(self.Ns - 1)])
        spot = np.array([(i + 1) * self.ds for i in range(self.Ns - 1)])
        strike = np.array([float(self.K) for i in range(self.Ns - 1)])

        V = np.array(zero <= (strike - spot), dtype=int)
        V_0 = np.array( [math.exp(-self.r*self.dt*n) for n in range(self.Nt+1)])

        #construct the matrix
        ai = np.array([(self.v**2*i**2/4 - self.r*i/4) for i in range(self.Ns+1)])
        bi = np.array([(self.v**2*i**2/2 + self.r/2) for i in range(self.Ns+1)])
        ci = np.array([(self.v**2*i**2/4 + self.r*i/4) for i in range(self.Ns+1)])
        sdt = np.array([-1.0/self.dt for i in range(self.Ns-1)])
        #A = sparse.spdiags([ai[2:], sdt[:]-bi[1:self.Ns], ci[:self.Ns-1]], [-1,0,1], self.Ns-1, self.Ns-1).toarray()
        B = sparse.spdiags([-ai[2:], sdt[:]+bi[1:self.Ns], -ci[:self.Ns-1]], [-1,0,1], self.Ns-1, self.Ns-1).toarray()
        AB = np.asarray([ci[:self.Ns-1],sdt[:]-bi[1:self.Ns],np.insert(ai,self.Ns,0)[2:self.Ns+1]])

        F0 = -ai[1]*(V_0[:self.Nt]+V_0[1:])

        for i in range(self.Nt):
            w = B.dot(V)
            w[0] = w[0] + F0[i]
            V = linalg.solve_banded((1,1),AB, w)

        return V[int(round(self.S0/self.ds))-1]

    def implicit_finite_call(self):
        # construct the time T payoff series
        zero = np.array([0.0 for i in range(self.Ns - 1)])
        spot = np.array([(i + 1) * self.ds for i in range(self.Ns - 1)])
        strike = np.array([float(self.K) for i in range(self.Ns - 1)])

        V = np.array(zero <= (spot - strike), dtype=int)
        V_Ns = np.array([math.exp(-self.r * self.dt * n) for n in range(self.Nt + 1)])

        # construct the matrix
        ai = np.array([(self.v ** 2 * i ** 2 / 4 - self.r * i / 4) for i in range(self.Ns + 1)])
        bi = np.array([(self.v ** 2 * i ** 2 / 2 + self.r / 2) for i in range(self.Ns + 1)])
        ci = np.array([(self.v ** 2 * i ** 2 / 4 + self.r * i / 4) for i in range(self.Ns + 1)])
        sdt = np.array([-1.0 / self.dt for i in range(self.Ns - 1)])
        # A = sparse.spdiags([ai[2:], sdt[:]-bi[1:self.Ns], ci[:self.Ns-1]], [-1,0,1], self.Ns-1, self.Ns-1).toarray()
        B = sparse.spdiags([-ai[2:], sdt[:] + bi[1:self.Ns], -ci[:self.Ns - 1]], [-1, 0, 1], self.Ns - 1,
                           self.Ns - 1).toarray()
        AB = np.asarray([ci[:self.Ns - 1], sdt[:] - bi[1:self.Ns], np.insert(ai, self.Ns, 0)[2:self.Ns + 1]])

        F_Nt = -ci[self.Ns-1 ] * (V_Ns[:self.Nt] + V_Ns[1:])

        for i in range(self.Nt):
            w = B.dot(V)
            w[self.Ns-2] = w[self.Ns-2] + F_Nt[self.Nt -1- i]
            V = linalg.solve_banded((1, 1), AB, w)

        return V[int(round(self.S0 / self.ds)) - 1]

    def price(self):
        if self.type == 'C':
            price = self.implicit_finite_call()
        elif self.type == 'P':
            price = self.implicit_finite_put()
        else:
            raise ValueError("The option type is wrong")
        return price

def main():
    #a = Implicit_EuDigi('C',0.03, 0.2, 1, 100, 300, 200, 200, 100)
    #b = Implicit_EuDigi('P',0.03, 0.2, 1, 100, 300, 1000, 10000, 100)
    #print('Call:', a.price())
    #print('Put:', b.price())

    #x = np.arange(100,5000,100)
    #for put option
    #y = np.asarray([(Implicit_EuDigi('P',0.03, 0.2, 1, 100, 300, x[i], x[i], 100).price() - 0.465873242) for i in range(len(x))])
    #for call option
    #y = np.asarray([(Implicit_EuDigi('C', 0.05, 0.3, 0.5, 40, 120, 81, 81, 40).price() - 0.49224034731308075) for i in range(len(x))])
    #plt.plot(x,y)
    #plt.title("Error change of digital call option")
    #plt.xlabel('Nt(Ns)')
    #plt.ylabel('Error')
    #plt.show()
    print(Implicit_EuDigi('C', 0.05, 0.3, 0.5, 40, 110, 10000, 641, 40).price() )


if __name__ == "__main__":main()

