# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 20:22:13 2023

@author: Vince
"""

import sympy as sp
import numpy as np


###########Debug Code for checking equations of motion
sp.init_printing(use_unicode=True)
sp.init_printing(pretty_print=False)
M = sp.symbols('M')
m = sp.symbols('m_1')
l = sp.symbols('l')
d = sp.symbols('d')
Ipsi = sp.symbols('I_psi')
Ithe = sp.symbols('I_theta')
Iphi = sp.symbols('I_phi')
Ip = sp.symbols('I_p')
g = sp.symbols('g')
q = []
q.append(sp.symbols('x'))
q.append(sp.symbols('y'))
q.append(sp.symbols('z'))
q.append(sp.symbols('psi'))
q.append(sp.symbols('theta'))
q.append(sp.symbols('phi'))
q.append(sp.symbols('alpha'))
q.append(sp.symbols('beta'))

q.append(sp.symbols('xdot'))
q.append(sp.symbols('ydot'))
q.append(sp.symbols('zdot'))
q.append(sp.symbols('psidot'))
q.append(sp.symbols('thetadot'))
q.append(sp.symbols('phidot'))
q.append(sp.symbols('alphadot'))
q.append(sp.symbols('betadot'))

def s(inpt):
    return sp.sin(inpt)
def c(inpt):
    return sp.cos(inpt)

m11 = M+m
m17 =  m*l*c(q[6])*c(q[7])
m18 =  -m*l*s(q[6])*s(q[7])
m27 =  m*l*c(q[6])*s(q[7])
m28 =  m*l*s(q[6])*c(q[7])
m37 =  m*l*s(q[6])
m44 =  Ipsi*s(q[4])**2+c(q[4])**2*(Ithe*s(q[5])**2+Iphi*c(q[5])**2)
m45 =  (Ithe - Iphi)*(c(q[4])*s(q[5])*c(q[5]))
m55 =  Ithe*c(q[5])**2+Iphi*s(q[5])**2
m77 = m*l**2+Ip
m88 = m*l**2*s(q[6])**2+Ip

M_q =  np.array([[m11,0,0,0,0,0,m17(q),m18(q)],
                          [0,m11,0,0,0,0,m27(q),m28(q)],
                          [0,0,m11,0,0,0,m37(q),0],
                          [0,0,0,m44(q),m45(q),-Ipsi*s(q[4]),0,0],
                          [0,0,0,m45(q),m55(q),0,0,0],
                          [0,0,0,-Ipsi*s(q[4]),0,Ipsi,0,0],
                          [m17(q),m27(q),m37(q),0,0,0,m77,0],
                          [m18(q),m28(q),0,0,0,0,0,m88(q)]])

c17 =  -m*l*(c(q[6])*s(q[7])*q[15]+s(q[6])*c(q[7])*q[14])
c18 =  -m*l*(c(q[6])*s(q[7])*q[14]+s(q[6])*c(q[7])*q[15])
c27 =  m*l*(c(q[6])*c(q[7])*q[15]-s(q[6])*s(q[7])*q[14])
c28 =  m*l*(c(q[6])*c(q[7])*q[14]-s(q[6])*s(q[7])*q[15])
c44 =  Ipsi*q[12]*s(q[4])*c(q[4])-(Ithe+Iphi)*(q[12]*s(q[4])*c(q[4])*s(q[5])**2)+\
                (Ithe-Iphi)*q[13]*c(q[4])**2*s(q[5])*c(q[5])
c45 =  Ipsi*q[11]*s(q[4])*c(q[4])-(Ithe-Iphi)*(q[12]*s(q[4])*c(q[5])*s(q[5])+q[13]*c(q[4])*s(q[5])**2)\
                -(Ithe+Iphi)*(q[11]*s(q[4])*c(q[4])*c(q[5])**2-q[13]*c(q[4])*c(q[5])**2)

c46 =  -(Ipsi*q[12]*c(q[4])-(Ithe-Iphi)*(q[11]*c(q[4])**2*s(q[5])*c(q[5])))
c54 =  q[11]*s(q[4])*c(q[4])*(-Ipsi+Ithe*s(q[5])**2+Iphi*c(q[5])**2)
c55 =  -(Ithe-Iphi)*(q[13]*s(q[5])*c(q[5]))
c56 =  Ipsi*q[11]*c(q[4])+(Ithe-Iphi)*(-q[12]*s(q[4])*c(q[5])+q[11]*c(q[4])*c(q[5])**2-q[11]*c(q[4])*s(q[5])**2)
c64 =  -(Ithe-Iphi)*(q[11]*c(q[4])**2*s(q[5])*c(q[5]))
c65 =  -Ipsi*q[11]*c(q[4])+(Ithe-Iphi)*(q[12]*s(q[5])*c(q[5])+q[11]*c(q[4])*s(q[5])**2-q[11]*c(q[4])*c(q[5])**2)

C_q =  np.array([[0,0,0,0,0,0,c17(q),c18(q)],
                          [0,0,0,0,0,0,c27(q),c28(q)],
                          [0,0,0,0,0,0,m*l*c(q[6])*q[14],0],
                          [0,0,0,c44(q),c45(q),c46(q),0,0],
                          [0,0,0,c54(q),c55(q),c56(q),0,0],
                          [0,0,0,c64(q),c65(q),0,0,0],
                          [0,0,0,0,0,0,0,-m*l**2*s(q[6])*c(q[6])*q[15]],
                          [0,0,0,0,0,0,m*l**2*s(q[6])*c(q[6])*q[15],m*l**2*s(q[6])*c(q[6])*q[14]]])

G_q =  np.array([[0],
                          [0],
                          [(M+m)*g],
                          [0],
                          [0],
                          [0],
                          [m*l*g*s(q[6])],
                          [0]])

b_q = np.array([[s(q[6])*s(q[3])+c(q[5])*c(q[3])*s(q[4]),0,0,0],
                          [c(q[5])*s(q[4])*s(q[3])-c(q[3])*s(q[5]),0,0,0],
                          [c(q[4])*c(q[5]),0,0,0],
                          [0,1,0,0],
                          [0,0,1,0],
                          [0,0,0,1],
                          [0,0,0,0],
                          [0,0,0,0]])
###########################################End Debug Code
