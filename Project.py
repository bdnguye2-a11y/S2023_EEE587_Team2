#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 26 15:43:28 2023

@author: vince
"""

import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from tqdm import *
import scipy

#              x,y,z  ,s,t,f,a,b,x,y,z,s,t,f,a   ,b

class NL_QUADCOPTER:
    def __init__(self):
        self.M_q = None
        self.C_q = None
        self.G_q = None
        self.b_q = None
        self.Ipsi = 1
        self.Ithe = 1
        self.Iphi = 1
        self.Ip = 0.1
        self.M = 0.5
        self.m = 0.5
        self.l = 1
        self.g = 9.91

    def set_param(self, Ipsi, Ithe, Iphi, Ip, M, m, l, g):
        self.Ipsi = Ipsi
        self.Ithe = Ithe
        self.Iphi = Iphi
        self.Ip = Ip
        self.M = M
        self.m = m
        self.l = l
        self.g = g

    def model(self):
        s = lambda angle: np.sin(angle)
        c = lambda angle: np.cos(angle)

        J = lambda q: np.array([[self.Ipsi * s(q[4]) ** 2 + self.Ithe * c(q[4]) ** 2 * s(q[5]) ** 2 + self.Iphi * c(self.Ithe) ** 2 * c(
                                q[5]) ** 2, c(q[4]) * c(q[5]) * s(q[5]) * (self.Ithe - self.Iphi), -self.Ipsi * s(q[4])],
                                [c(q[4]) * c(q[5]) * s(q[5]) * (self.Ithe - self.Iphi), self.Ithe * c(q[5]) ** 2 + self.Iphi * s(q[5]) ** 2,
                                 0],
                                [-self.Iphi * s(q[4]), 0, self.Ipsi],
                                ])

        m11 = self.M + self.m
        m17 = lambda q: self.m * self.l * c(q[6]) * c(q[7])
        m18 = lambda q: -self.m * self.l * s(q[6]) * s(q[7])
        m27 = lambda q: self.m * self.l * c(q[6]) * s(q[7])
        m28 = lambda q: self.m * self.l * s(q[6]) * c(q[7])
        m37 = lambda q: self.m * self.l * s(q[6])
        m44 = lambda q: self.Ipsi * s(q[4]) ** 2 + c(q[4]) ** 2 * (self.Ithe * s(q[5]) ** 2 + self.Iphi * c(q[5]) ** 2)
        m45 = lambda q: (self.Ithe - self.Iphi) * (c(q[4]) * s(q[5]) * c(q[5]))
        m55 = lambda q: self.Ithe * c(q[5]) ** 2 + self.Iphi * s(q[5]) ** 2
        m77 = self.m * self.l ** 2 + self.Ip
        m88 = lambda q: self.m * self.l ** 2 * s(q[6]) ** 2 + self.Ip

        self.M_q = lambda q: np.array([[m11, 0, 0, 0, 0, 0, m17(q), m18(q)],
                                       [0, m11, 0, 0, 0, 0, m27(q), m28(q)],
                                       [0, 0, m11, 0, 0, 0, m37(q), 0],
                                       [0, 0, 0, m44(q), m45(q), -Ipsi * s(q[4]), 0, 0],
                                       [0, 0, 0, m45(q), m55(q), 0, 0, 0],
                                       [0, 0, 0, -Ipsi * s(q[4]), 0, Ipsi, 0, 0],
                                       [m17(q), m27(q), m37(q), 0, 0, 0, m77, 0],
                                       [m18(q), m28(q), 0, 0, 0, 0, 0, m88(q)]])

        c17 = lambda q: -self.m * self.l * (c(q[6]) * s(q[7]) * q[15] + s(q[6]) * c(q[7]) * q[14])
        c18 = lambda q: -self.m * self.l * (c(q[6]) * s(q[7]) * q[14] + s(q[6]) * c(q[7]) * q[15])
        c27 = lambda q: self.m * self.l * (c(q[6]) * c(q[7]) * q[15] - s(q[6]) * s(q[7]) * q[14])
        c28 = lambda q: self.m * self.l * (c(q[6]) * c(q[7]) * q[14] - s(q[6]) * s(q[7]) * q[15])
        c44 = lambda q: self.Ipsi * q[12] * s(q[4]) * c(q[4]) - (self.Ithe + self.Iphi) * (q[12] * s(q[4]) * c(q[4]) * s(q[5]) ** 2) + \
                        (self.Ithe - self.Iphi) * q[13] * c(q[4]) ** 2 * s(q[5]) * c(q[5])
        c45 = lambda q: self.Ipsi * q[11] * s(q[4]) * c(q[4]) - (self.Ithe - self.Iphi) * (
                    q[12] * s(q[4]) * c(q[5]) * s(q[5]) + q[13] * c(q[4]) * s(q[5]) ** 2) \
                        - (self.Ithe + self.Iphi) * (q[11] * s(q[4]) * c(q[4]) * c(q[5]) ** 2 - q[13] * c(q[4]) * c(q[5]) ** 2)

        c46 = lambda q: -(self.Ipsi * q[12] * c(q[4]) - (self.Ithe - self.Iphi) * (q[11] * c(q[4]) ** 2 * s(q[5]) * c(q[5])))
        c54 = lambda q: q[11] * s(q[4]) * c(q[4]) * (-self.Ipsi + self.Ithe * s(q[5]) ** 2 + self.Iphi * c(q[5]) ** 2)
        c55 = lambda q: -(self.Ithe - self.Iphi) * (q[13] * s(q[5]) * c(q[5]))
        c56 = lambda q: self.Ipsi * q[11] * c(q[4]) + (self.Ithe - self.Iphi) * (
                    -q[12] * s(q[4]) * c(q[5]) + q[11] * c(q[4]) * c(q[5]) ** 2 - q[11] * c(q[4]) * s(q[5]) ** 2)
        c64 = lambda q: -(self.Ithe - self.Iphi) * (q[11] * c(q[4]) ** 2 * s(q[5]) * c(q[5]))
        c65 = lambda q: -self.Ipsi * q[11] * c(q[4]) + (self.Ithe - self.Iphi) * (
                    q[12] * s(q[5]) * c(q[5]) + q[11] * c(q[4]) * s(q[5]) ** 2 - q[11] * c(q[4]) * c(q[5]) ** 2)

        self.C_q = lambda q: np.array([[0, 0, 0, 0, 0, 0, c17(q), c18(q)],
                                       [0, 0, 0, 0, 0, 0, c27(q), c28(q)],
                                       [0, 0, 0, 0, 0, 0, self.m * self.l * c(q[6]) * q[14], 0],
                                       [0, 0, 0, c44(q), c45(q), c46(q), 0, 0],
                                       [0, 0, 0, c54(q), c55(q), c56(q), 0, 0],
                                       [0, 0, 0, c64(q), c65(q), 0, 0, 0],
                                       [0, 0, 0, 0, 0, 0, 0, -self.m * self.l ** 2 * s(q[6]) * c(q[6]) * q[15]],
                                       [0, 0, 0, 0, 0, 0, self.m * self.l ** 2 * s(q[6]) * c(q[6]) * q[15],
                                        self.m * self.l ** 2 * s(q[6]) * c(q[6]) * q[14]]])

        self.G_q = lambda q: np.array([[0],
                                       [0],
                                       [(self.M + self.m) * self.g],
                                       [0],
                                       [0],
                                       [0],
                                       [self.m * self.l * self.g * s(q[6])],
                                       [0]])

        self.b_q = lambda q: np.array([[s(q[6]) * s(q[3]) + c(q[5]) * c(q[3]) * s(q[4]), 0, 0, 0],
                                       [c(q[5]) * s(q[4]) * s(q[3]) - c(q[3]) * s(q[5]), 0, 0, 0],
                                       [c(q[4]) * c(q[5]), 0, 0, 0],
                                       [0, 1, 0, 0],
                                       [0, 0, 1, 0],
                                       [0, 0, 0, 1],
                                       [0, 0, 0, 0],
                                       [0, 0, 0, 0]])

    def derive(self, inq, inU, noise=0):
        if noise != 0:
            self.model()
            temp = np.zeros(16)
            temp[:8] = inq[8:]
            temp[8:] = (np.linalg.inv(self.M_q(inq.squeeze())) @ (
                        self.b_q(inq.squeeze()) @ inU - self.C_q(inq.squeeze()) @ inq[8:].reshape(8, 1) - self.G_q(
                    inq.squeeze()))).squeeze()
            return temp

    def evolve(self, t, q0, u):
        y = scipy.integrate.odeint(self.derive, q0, t, args=(u,))
        return y


if __name__ == "__main__":
    sim = NL_QUADCOPTER()
    q0 = np.array([0, 0, 2.0, 0, 0, 0, np.pi / 4, 0, 0, 0, 0, 0, 0, 0, 0, .1])
    u = [0, 0, 0, 0]
    dt = 0.01
    y = sim.evolve([dt], q0, u)
