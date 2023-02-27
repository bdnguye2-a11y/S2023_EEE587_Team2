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

M = 1
m = 1
d = 1
Ipsi = 1
Ithe = 1
Iphi = 1
Ip = 1
l = 1

#q = [x,y,z,psi,theta,phi,alpha,beta].T
#     0,1,2, 3 ,  4  , 5 ,  6  ,  7 
s = lambda angle: np.sin(angle)
c = lambda angle: np.cos(angle)

J = lambda Ipsi,Ithe,Iphi,q:np.array([[Ipsi*s(q[4])**2+Ithe*c(q[4])**2*s(q[5])**2+Iphi*c(the)**2*c(q[5])**2,c(q[4])*c(q[5])*s(q[5])*(Ithe-Iphi),-Ipsi*s(q[4])]
                     [c(q[4])*c(q[5])*s(q[5])*(Ithe-Iphi),Ithe*c(q[5])**2+Iphi*s(q[5])**2,0]
                     [-Iphi*s(q[4]),0,Ipsi]
                     ])

m11 = M+m
m17 = lambda q: m*l*c(q[6])*c(q[7])
m18 = lambda q: -m*l*s(q[6])*s(q[7])
m27 = lambda q: m*l*c(q[6])*s*(q[7])
m28 = lambda q: m*l*s(q[6])*c(q[7])
m37 = lambda q: m*l&s(q[6])
m44 = lambda q: Ipsi*s(q[4])**2+c(q[4])**2*(Ithe*s(q[5])**2+Iphi*c(q[5])**2)
m45 = lambda q: (Ithe - Iphi)*(c(q[4])*s(q[5])*c(q[5]))
m55 = lambda q: Ithe*c(q[5])**2+Iphi*s(q[5])**2
m77 = m*l**2+Ip
m88 = lambda q:m*l**2*s(q[6])**2+Ip

M_q = lambda q: np.array([[m11,0,0,0,0,0,m17(q),m18(q))]
                          [0,]
                          []
                          []
                          []
                          []
                          []
                          ])