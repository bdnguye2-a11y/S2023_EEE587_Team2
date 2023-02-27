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

M = .5
m = .1
d = .015
Ipsi = 1
Ithe = 1
Iphi = 1
Ip = .1
l = 1
g = 9.81

#q = [x,y,z,psi,theta,phi,alpha,beta].T
#     0,1,2, 3 ,  4  , 5 ,  6  ,  7 
#     8,9,10,11, 12  , 13, 14  , 15

q0 = np.array([0,0,2,0,0,0,0,0,0,0,0,0,0,0,0,0])

s = lambda angle: np.sin(angle)
c = lambda angle: np.cos(angle)

J = lambda q:np.array([[Ipsi*s(q[4])**2+Ithe*c(q[4])**2*s(q[5])**2+Iphi*c(the)**2*c(q[5])**2,c(q[4])*c(q[5])*s(q[5])*(Ithe-Iphi),-Ipsi*s(q[4])],
                     [c(q[4])*c(q[5])*s(q[5])*(Ithe-Iphi),Ithe*c(q[5])**2+Iphi*s(q[5])**2,0],
                     [-Iphi*s(q[4]),0,Ipsi],
                     ])

m11 = M+m
m17 = lambda q: m*l*c(q[6])*c(q[7])
m18 = lambda q: -m*l*s(q[6])*s(q[7])
m27 = lambda q: m*l*c(q[6])*s(q[7])
m28 = lambda q: m*l*s(q[6])*c(q[7])
m37 = lambda q: m*l*s(q[6])
m44 = lambda q: Ipsi*s(q[4])**2+c(q[4])**2*(Ithe*s(q[5])**2+Iphi*c(q[5])**2)
m45 = lambda q: (Ithe - Iphi)*(c(q[4])*s(q[5])*c(q[5]))
m55 = lambda q: Ithe*c(q[5])**2+Iphi*s(q[5])**2
m77 = m*l**2+Ip
m88 = lambda q:m*l**2*s(q[6])**2+Ip

M_q = lambda q: np.array([[m11,0,0,0,0,0,m17(q),m18(q)],
                          [0,m11,0,0,0,0,m27(q),m28(q)],
                          [0,0,m11,0,0,0,m37(q),0],
                          [0,0,0,m44(q),m45(q),-Iphi*s(q[4]),0,0],
                          [0,0,0,m45(q),m55(q),0,0,0],
                          [0,0,0,-Ipsi*s(q[4]),0,Ipsi,0,0],
                          [m17(q),m27(q),m37(q),0,0,0,m77,0],
                          [m18(q),m28(q),0,0,0,0,0,m88(q)]])

c17 = lambda q: -m*l*(c(q[6])*s(q[7])*q[15]+s(q[6])*c(q[7])*q[14])
c18 = lambda q: -m*l*(c(q[6])*s(q[7])*q[14]+s(q[6])*c(q[7])*q[15])
c27 = lambda q: m*l*(c(q[6])*c(q[7])*q[15]-s(q[6])*s(q[7])*q[14])
c28 = lambda q: m*l*(c(q[6])*c(q[7])*q[14]-s(q[6])*s(q[7])*q[15])
c44 = lambda q: Ipsi*q[12]*s(q[4])*c(q[4])-(Ithe+Iphi)*(q[12]*s(q[4])*c(q[4])*s(q[5])**2)+\
                (Ithe-Iphi)*q[13]*c(q[4])**2*s(q[5])*c(q[5])
c45 = lambda q: Ipsi*q[11]*s(q[4])*c(q[4])-(Ithe-Iphi)*(q[12]*s(q[4])*c(q[5])*s(q[5])+q[13]*c(q[4])*s(q[5])**2)\
                +(Ithe+Iphi)*(q[11]*s(q[4])*c(q[4])*c(q[5])**2-q[13]*c(q[4])*c(q[5])**2)

c46 = lambda q: -(Ipsi*q[12]*c(q[4])-(Ithe-Iphi)*(q[11]*c(q[4])**2*s(q[5])*c(q[5])))
c54 = lambda q: q[11]*s(q[4])*c(q[4])*(-Ipsi+Ithe*s(q[5])**2+Iphi*c(q[5])**2)
c55 = lambda q: -(Ithe-Iphi)*(q[13]*s(q[5])*c(q[5]))
c56 = lambda q: Ipsi*q[11]*c(q[4])+(Ithe-Iphi)*(-q[12]*s(q[4])*c(q[5])+q[11]*c(q[4])*c(q[5])**2-q[11]*c(q[4])*s(q[5])**2)
c64 = lambda q: -(Ithe-Iphi)*(q[11]*c(q[4])**2*s(q[5])*c(q[5]))
c65 = lambda q: -Ipsi*q[11]*c(q[4])+(Ithe-Iphi)*(q[12]*s(q[5])*c(q[5])+q[11]*c(q[4])*s(q[5])**2-q[11]*c(q[4])*c(q[5])**2)

C_q = lambda q: np.array([[0,0,0,0,0,0,c17(q),c18(q)],
                          [0,0,0,0,0,0,c27(q),c28(q)],
                          [0,0,0,0,0,0,m*l*c(q[6])*q[14],0],
                          [0,0,0,c44(q),c45(q),c46(q),0,0],
                          [0,0,0,c54(q),c55(q),c56(q),0,0],
                          [0,0,0,c64(q),c65(q),0,0,0],
                          [0,0,0,0,0,0,0,-m*l**2*s(q[6])*c(q[6])*q[15]],
                          [0,0,0,0,0,0,m*l**2*s(q[6])*c(q[6])*q[15],m*l**2*s(q[6])*c(q[6])*q[14]]])

G_q = lambda q: np.array([[0],
                          [0],
                          [(M+m)*g],
                          [0],
                          [0],
                          [0],
                          [m*l*g*s(q[6])],
                          [0]])

b_q = lambda q:np.array([[s(alpha)*s(psi)+c(phi)*c(psi)*s(theta),0,0,0],
                         [c(phi)*s(theta)*s(psi)-c(psi)*s(phi),0,0,0],
                         [c(theta)*c(phi),0,0,0],
                         [0,1,0,0],
                         [0,0,1,0],
                         [0,0,0,1],
                         [0,0,0,0],
                         [0,0,0,0]])