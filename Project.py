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

M = .5
m = .5
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

#              x,y,z  ,s,t,f,a,b,x,y,z,s,t,f,a   ,b
q0 = np.array([0,0,2.0,0,0,0,np.pi/4,0,0,0,0,0,0,0,0,.1])

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
                          [0,0,0,m44(q),m45(q),-Ipsi*s(q[4]),0,0],
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
                -(Ithe+Iphi)*(q[11]*s(q[4])*c(q[4])*c(q[5])**2-q[13]*c(q[4])*c(q[5])**2)

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

b_q = lambda q:np.array([[s(q[6])*s(q[3])+c(q[5])*c(q[3])*s(q[4]),0,0,0],
                         [c(q[5])*s(q[4])*s(q[3])-c(q[3])*s(q[5]),0,0,0],
                         [c(q[4])*c(q[5]),0,0,0],
                         [0,1,0,0],
                         [0,0,1,0],
                         [0,0,0,1],
                         [0,0,0,0],
                         [0,0,0,0]])

U = np.array([[9.81],
              [0],
              [0],
              [0]])

q = q0.copy()

def evolve(inq,innt,inU=U,noise=0):
    if noise != 0:
        
    temp = np.zeros(16)
    temp[:8] = inq[8:]
    temp[8:]=(np.linalg.inv(M_q(inq.squeeze()))@(b_q(inq.squeeze())@inU - C_q(inq.squeeze())@inq[8:].reshape(8,1)-G_q(inq.squeeze()))).squeeze()
    return temp

t = np.linspace(0,4,10000)
y = scipy.integrate.odeint(evolve,q0,t,args=(U,t))

# dt = .001
# time = np.linspace(0,10,int(10/dt)+1)
# steps = 20000
# z = np.zeros((steps,16))
# qd = q0[8:].copy().reshape(8,1)
# qn = q0[:8].copy().reshape(8,1)
# histdd = []

# for i in trange(steps):
#     # if i==2500:
#     #     U[0,0] = 0
#     qdd = np.linalg.inv(M_q(q.squeeze()))@(b_q(q.squeeze())@U - C_q(q.squeeze())@q[8:].reshape(8,1)-G_q(q.squeeze()))
#     histdd.append(qdd)
#     qd += 0.5*dt*(qdd+q[8:].reshape(8,1))
#     qn += 0.5*dt*(qd+q[:8].reshape(8,1))
#     q = np.concatenate((qn,qd))
#     z[i] = q.T.copy()

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.plot(y[:,0],y[:,1],y[:,2])

# histdd = np.array(histdd).squeeze()

fig = plt.figure()
plt.grid()
plt.plot(y[:,0])

fig = plt.figure()
plt.grid()
plt.plot(y[:,2])

dt = .001
time = np.linspace(0,10,int(10/dt)+1) # not used... Why is there a steps value? For the loading bar?
steps = 5000
z = np.zeros((steps,16))
qd = q0[8:].copy().reshape(8,1) # first derivative ?
qn = q0[:8].copy().reshape(8,1) # position values ?

for i in trange(steps):
    qdd = np.linalg.inv(M_q(q.squeeze()))@(b_q(q.squeeze())@U - C_q(q.squeeze())@q[8:].reshape(8,1)-G_q(q.squeeze()))

    # where does 0.5*dt(...) come from? should it be qd += qdd*dt instead?
    qd += 0.5*dt*(qdd+q[8:].reshape(8,1))
    qn += 0.5*dt*(qd+q[:8].reshape(8,1))
    q = np.concatenate((qn,qd))
    z[i] = q.T

for i in range(8):
    plt.plot(z[:,i])
    plt.show()

