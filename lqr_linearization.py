# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 21:10:11 2023

@author: Vince
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import sympy as sp
from sympy.physics.mechanics import dynamicsymbols
from tqdm import *
import control as ctl

# sp.init_printing(use_unicode=True)
sp.init_printing(pretty_print=False)


def calc(inA,inB,inC):
    
    temp = inA
    tempn = len(inB)
    # print(tempn)
    # print(len(inB))
    # print(len(inC))
    # print('')
    for i in range(tempn):
        temp = temp.subs(inB[tempn-i-1],inC[tempn-i-1])
    return temp

def multi_calc(inA,inB,inC,pbr=False):
    tempn1 = len(inA)
    tempn2 = len(inB)
    
    temp = inA.copy()
    
    if pbr:
        with tqdm(total=tempn1*tempn2) as pbar:
            for i in range(tempn1):
                for j in range(tempn2):
                    temp[i] = calc(temp[i],inB,inC)
                    pbar.update(1)
    else:
        for i in range(tempn1):
            for j in range(tempn2):
                temp[i] = calc(temp[i],inB,inC)
    return temp


print('Initializing Variables...')
c = lambda x: sp.cos(x)
s = lambda x: sp.sin(x)
syms = []

syms.append(dynamicsymbols('x'))
syms.append(sp.diff(syms[-1]))
syms.append(dynamicsymbols('y'))
syms.append(sp.diff(syms[-1]))
syms.append(dynamicsymbols('z'))
syms.append(sp.diff(syms[-1]))

syms.append(dynamicsymbols('psi'))
syms.append(sp.diff(syms[-1]))
syms.append(dynamicsymbols('theta'))
syms.append(sp.diff(syms[-1]))
syms.append(dynamicsymbols('phi'))
syms.append(sp.diff(syms[-1]))

syms.append(dynamicsymbols('alpha'))
syms.append(sp.diff(syms[-1]))
syms.append(dynamicsymbols('beta'))
syms.append(sp.diff(syms[-1]))

t = sp.symbols('t')

x = syms[0]
xdot = syms[1]
y = syms[2]
ydot = syms[3]
z = syms[4]
zdot = syms[5]

psi = syms[6]
psidot = syms[7]
theta = syms[8]
thetadot = syms[9]
phi = syms[10]
phidot = syms[11]

alpha = syms[12]
alphadot = syms[13]
beta = syms[14]
betadot = syms[15]

q = sp.Matrix([x,y,z,psi,theta,phi,alpha,beta])

m = sp.symbols('m')
l = sp.symbols('l')
M = sp.symbols('M')
g = sp.symbols('g')

I_psi = sp.symbols('I_psi')
I_theta = sp.symbols('I_theta')
I_phi = sp.symbols('I_phi')

I_p = sp.symbols('I_p')


zeta = sp.Matrix([x,y,z])
r = sp.Matrix([s(alpha)*c(beta),s(alpha)*s(beta),-c(alpha)])
zetap = zeta+l*r
eta = sp.Matrix([psi,theta,phi])

J = sp.zeros(3,3)

s_th = s(theta)
c_th = c(theta)
c_ph = c(phi)
s_ph = s(phi)

J[0,0] = I_psi*s_th**2 + I_theta*c_th**2*s_ph**2+I_phi*c_th**2*c_ph**2
J[0,1] = c_th*c_ph*s_ph*(I_theta-I_phi)
J[1,0] = J[0,1]
J[1,1] = I_theta*c_ph**2+I_phi*s_ph**2
J[0,2] = -I_psi*s_th
J[2,0] = J[0,2]
#J[1,2] = 0
J[2,2] = I_psi

zetadot = sp.diff(zeta,t)
etadot = sp.diff(eta,t)

zetapdot = sp.diff(zetap,t)

m11 = M+m
m22 = m11
m33 = m11
m17 = m*l*c(alpha)*c(beta)
m71 = m17
m18 = -m*l*s(alpha)*s(beta)
m81 = m18
m27 = m*l*c(alpha)*s(beta)
m72 = m27
m28 = m*l*s(alpha)*c(beta)
m82 = m28
m37 = m*l*s(alpha)
m73 = m37
m44 = I_psi*s(theta)**2+c(theta)**2*(I_theta*s(phi)**2+I_phi*c(phi)**2)
m45 = (I_theta-I_phi)*(c(theta)*s(phi)*c(phi))
m54 = m45
m55 = I_theta*c(phi)**2+I_phi*s(phi)**2
m77 = m*l**2+I_p
m88 = m*l**2*s(alpha)**2+I_p

M_q = sp.Matrix([[m11,0,0,0,0,0,m17,m18],
                 [0,m22,0,0,0,0,m27,m28],
                 [0,0,m33,0,0,0,m37,0],
                 [0,0,0,m44,m45,-I_psi*s(theta),0,0],
                 [0,0,0,m54,m55,0,0,0],
                 [0,0,0,-I_psi*s(theta),0,I_psi,0,0],
                 [m71,m72,m73,0,0,0,m77,0],
                 [m81,m82,0,0,0,0,0,m88]])

c17 = -m*l*(c(alpha)*s(beta)*betadot+s(alpha)*c(beta)*alphadot)
c18 = -m*l*(c(alpha)*s(beta)*alphadot+s(alpha)*c(beta)*betadot)
c27 = m*l*(c(alpha)*c(beta)*betadot-s(alpha)*s(beta)*alphadot)
c28 = m*l*(c(alpha)*c(beta)*alphadot-s(alpha)*s(beta)*betadot)
c44 = I_psi*thetadot*s(theta)*c(theta)-(I_theta+I_phi)*(thetadot*s(theta)*c(theta)*s(phi)**2)+(I_theta-I_phi)*phidot*c(theta)**2*s(phi)*c(phi)
c45 = I_psi*psidot*s(theta)*c(theta)-(I_theta-I_phi)*(thetadot*s(theta)*c(phi)*s(phi)+phidot*c(theta)*s(phi)**2)-(I_theta+I_phi)*(psidot*s(theta)*c(theta)*c(phi)**2-phidot*c(theta)*c(phi)**2)
c46 = -(I_psi*thetadot*c(theta)-(I_theta-I_phi)*(psidot*c(theta)**2*s(phi)*c(phi)))
c54 = psidot*s(theta)*c(theta)*(-I_psi+I_theta*s(phi)**2+I_phi*c(phi)**2)
c55 = -(I_theta-I_phi)*(phidot*s(phi)*c(phi))
c56 = I_psi*psidot*c(theta)+(I_theta-I_phi)*(-thetadot*s(theta)*c(phi)+psidot*c(theta)*c(phi)**2-psidot*c(theta)*s(phi)**2)
c64 = -(I_theta-I_phi)*(psidot*c(theta)**2*s(phi)*c(phi))
c65 = -I_psi*psidot*c(theta)+(I_theta-I_phi)*(thetadot*s(phi)*c(phi)+psidot*c(theta)*s(phi)**2-psidot*c(theta)*c(phi)**2)

C_q = sp.Matrix([[0,0,0,0,0,0,c17,c18],
                 [0,0,0,0,0,0,c27,c28],
                 [0,0,0,0,0,0,m*c(alpha)*alphadot,0],
                 [0,0,0,c44,c45,c46,0,0],
                 [0,0,0,c54,c55,c56,0,0],
                 [0,0,0,c64,c65,0,0,0],
                 [0,0,0,0,0,0,0,-m*l**2*s(alpha)*c(alpha)*betadot],
                 [0,0,0,0,0,0,m*l**2*s(alpha)*c(alpha)*betadot,m*l**2*s(alpha)*c(alpha)*alphadot]])

G_q = sp.Matrix([0,0,(M+m)*g,0,0,0,m*l*g*s(alpha),0])

b = sp.zeros(8,4)

b[0,0] = s(alpha)*s(psi)+c(phi)*c(psi)*s(theta)
b[1,0] = c(phi)*s(theta)*s(psi)-c(psi)*s(phi)
b[2,0] = c(theta)*c(phi)
b[3,1]=1
b[4,2]=1
b[5,3]=1

qdot = sp.diff(q,t)
qddot = sp.diff(qdot,t)

inpts = sp.symbols('u1:5')

u = sp.Matrix([inpts[0],inpts[1],inpts[2],inpts[3]])

U = b*u

X1 = q
X2 = qdot

X = sp.Matrix([X1,X2])
F1 = X2
Z = M_q**-1
N = -Z*(C_q*X2+G_q)
F2 = N+Z*U

Xdot = sp.Matrix([F1,F2])

xsbls = sp.symbols('x1:17')

X_dot = multi_calc(Xdot,X[:,0],xsbls,True)

params = [m,l,M,g,I_psi,I_theta,I_phi,I_p]
vals = [.3,.242,.56,9.81,.0021,.006178,.006178,.56*.242**2]

X_dot = multi_calc(X_dot,params,vals)

A_ = sp.zeros(16)
B_ = sp.zeros(16,4)

with tqdm(total = 16*16+16*4) as pbar:
    for row in range(16):
        for column in range(16):
            A_[row,column] = sp.diff(X_dot[row],xsbls[column])
            pbar.update(1)
    for row in range(16):
        for column in range(4):
            B_[row,column] = sp.diff(X_dot[row],u[column])
            pbar.update(1)

def linearize(inq,inu):
    tempA = np.zeros((16,16))
    tempB = np.zeros((16,4))
    
    for row in range(16):
        for column in range(16):
            tempA[row,column] = calc(calc(A_[row,column], xsbls, inq.squeeze()),u[:,0],inu.squeeze())
        for column in range(4):
            tempB[row,column] = calc(calc(B_[row,column], xsbls, inq.squeeze()),u[:,0],inu.squeeze())
    return tempA,tempB


x0 = np.array([[-1],
                [-1],
                [-1],
                [0],
                [0],
                [0],
                [0],
                [0],
               
                [-.2],
                [.3],
                [.1],
                [0],
                [0],
                [0],
                [.02],
                [0]])

xf = np.array([[0],
                [0],
                [0],
                [0],
                [0],
                [0],
                [0],
                [0],
               
                [0],
                [0],
                [0],
                [0],
                [0],
                [0],
                [0],
                [0]])

u0 = np.array([10,.001,.001,.001])

A,B = linearize(x0,u0)
C = np.array([0,0,0, 0,0,0, 1,0, 0,0,0, 0,0,0, 0,0,])


H = np.zeros((16,16))
H[0,0] = 1
H[1,1] = 1
H[2,2] = 1
H[8,8] = 1
H[9,9] = 1
H[10,10] = 1

Q = np.diag([0,0,0, 0,0,0, 0,0, 0,0,0, 0,0,0, 0,0])
R = np.diag([.1,.1,.1,.1])

P = [H]
F = []

tf = 5
num = 150

tinv = np.linspace(0,5,151)

def discretize(inq,inu):
    tempA,tempB = linearize(inq,inu)
    tempC = np.array([0,0,0, 0,0,0, 1,0, 0,0,0, 0,0,0, 0,0])
    sysd = ctl.c2d(ctl.ss(tempA,tempB,tempC,np.array([0,0,0,0])),tf/num)
    
    return sysd.A,sysd.B
    
Ad,Bd = discretize(x0, u0)

Ads = []
Bds = []

for i in range(num):
    Ads.append(Ad)
    Bds.append(Bd)

usts = []

for i in trange(num):
    F_calc = -np.linalg.inv(R+Bds[num-i-1].T@P[-1]@Bds[num-1-i])@Bds[num-1-i].T@P[-1]@Ads[num-1-i]
    F.append(F_calc)
    P_calc = (Ads[num-1-i]+Bds[num-1-i]@F[-1]).T@P[-1]@(Ads[num-1-i]+Bds[num-1-i]@F[-1])+F[-1].T@R@F[-1]+Q
    P.append(P_calc)

xst = x0
xsts = [xst]

usts = [ust]

for i in range(num):
    ust = F[-i-1]@xst
    xst = Ad@xst+Bd@ust
    xsts.append(xst)



for i in trange(1,num):
    xst = xsts[num-i]
    ust = F[i-1]@xst
    Ad,Bd = discretize(xst,ust)
    for j in range(num-i+1):
        Ads[j] = Ad
        Bds[j] = Bd
    
    for j in range(i,num):
        F_calc = -np.linalg.inv(R+Bds[j].T@P[j]@Bds[num-1-j])@Bds[num-1-j].T@P[-1]@Ads[num-1-j]
        F.append(F_calc)
        P_calc = (Ads[num-1-i]+Bds[num-1-i]@F[-1]).T@P[-1]@(Ads[num-1-i]+Bds[num-1-i]@F[-1])+F[-1].T@R@F[-1]+Q
        P[]

    xst = x0
    for j in range(num-i):
        xst = Ads[j]@xst+Bds[j]@F[-j-1]@xst
        xsts[j] = xst
    

xsts = np.array(xsts).squeeze()

xs = xsts[:,0]
ys = xsts[:,1]
zs = xsts[:,2]
als = xsts[:,6]

# plt.plot(xsts[:,0])
# plt.figure()
# plt.plot(xsts[:,1])
# plt.figure()
# plt.plot(xsts[:,2])
# plt.figure()
# plt.plot(xsts[:,3])
# plt.figure()
# plt.plot(xsts[:,4])
# plt.figure()
# plt.plot(xsts[:,5])
# plt.figure()
# plt.plot(xsts[:,6])
# plt.figure()
# plt.plot(xsts[:,7])
# plt.figure()

# plt.plot(xsts[:,8])
# plt.figure()
# plt.plot(xsts[:,9])
# plt.figure()
# plt.plot(xsts[:,10])
# plt.figure()
# plt.plot(xsts[:,11])
# plt.figure()
# plt.plot(xsts[:,12])
# plt.figure()
# plt.plot(xsts[:,13])
# plt.figure()
# plt.plot(xsts[:,14])
# plt.figure()
# plt.plot(xsts[:,15])
# plt.figure()



# F2 = []

# for j in trange(2,num):
    
#     xst = xsts[-j]
#     ust = F[-j+1]@xst
#     Ad,Bd = discretize(xst,ust)
    
    
    
#     for i in trange(j,num):
#         F_calc = -np.linalg.inv(R+Bd.T@P2[-1]@Bd)@Bd.T@P2[-1]@Ad
        
#         F2.append(F_calc)
#         P_calc = (Ad+Bd@F[-1]).T@P2[-1]@(Ad+Bd@F2[-1])+F2[-1].T@R@F2[-1]+Q
#         P2.append(P_calc)
    
#     for i in trange(1,len(F)):
#         ust = F[-i]@xst
#         Ad,Bd = discretize(xst,ust)
#         Ads.append(Ad)
#         Bds.append(Bd)
#         xst = Ad@xst+Bd@ust
#         xsts2.append(xst)

# xsts2 = np.array(xsts2).squeeze()

# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')
# ax.plot(xsts2[:,0],xsts2[:,1],xsts2[:,2])
# ax.plot(xsts[:,0],xsts[:,1],xsts[:,2])

# plt.figure(1)
# plt.plot(tinv,als)

# ###
# xst = x0
# xs2 = [x0[0,0]]
# ys2 = [x0[1,0]]
# zs2 = [x0[2,0]]
# als2 = [0]
# for i in range(len(F)):
#     xst = Ad@xst+Bd@F[-i]@xst+np.random.normal(0,.01,(16,1))
#     xs2.append(xst[0,0])
#     ys2.append(xst[1,0])
#     zs2.append(xst[2,0])

#     als2.append(xst[6,0])

# # plt.figure(2)
# plt.plot(tinv,xs2)
# plt.plot(tinv,ys2)
# plt.plot(tinv,zs2)
# plt.legend([r'$\alpha$ no filter',r'$\alpha$ Kalman Filter'])
# plt.xlabel('Time (s)')
# plt.ylabel('Position (m)')
# plt.grid()
# plt.figure(1)
# plt.plot(tinv,als2)
# plt.legend(['Q6=10','Q6=1'])
# plt.grid()
# ###

Q = np.diag([1,1,1, 0,0,0, 10,0, 0,0,0, 0,0,0, 0,0])
R = np.diag([.1,.1,.1,.1])

P = [H]
F = []

xst = x0
ust = u0
for i in trange(num):
    F_calc = -np.linalg.inv(R+Bd.T@P[-1]@Bd)@Bd.T@P[-1]@Ad
    
    F.append(F_calc)
    P_calc = (Ad+Bd@F[-1]).T@P[-1]@(Ad+Bd@F[-1])+F[-1].T@R@F[-1]+Q
    P.append(P_calc)

xst = x0
xs2 = [x0[0,0]]
ys2 = [x0[1,0]]
zs2 = [x0[2,0]]
als2 = [0]
for i in range(len(F)):
    xst = Ad@xst+Bd@F[-i]@xst
    xs2.append(xst[0,0])
    ys2.append(xst[1,0])
    zs2.append(xst[2,0])

    als2.append(xst[6,0])

plt.figure(0)
plt.plot(tinv,xs)
plt.plot(tinv,xs2)
plt.legend(['Q=0','Q=10'])
plt.xlabel('Time (s)')
plt.ylabel('X Position (m)')
plt.grid()

plt.figure(1)
plt.plot(tinv,ys)
plt.plot(tinv,ys2)
plt.legend(['Q=0','Q=10'])
plt.xlabel('Time (s)')
plt.ylabel('Y Position (m)')
plt.grid()

plt.figure(2)

plt.plot(tinv,zs)
plt.plot(tinv,zs2)
plt.legend(['Q=0','Q=10'])
plt.xlabel('Time (s)')
plt.ylabel('Z Position (m)')
plt.grid()

plt.figure(3)
plt.plot(tinv,als)
plt.plot(tinv,als2)
plt.legend(['Q6=0','Q6=1'])
plt.xlabel('Time (s)')
plt.ylabel(r'Swing Angle $\alpha$ (deg)')
plt.grid()
