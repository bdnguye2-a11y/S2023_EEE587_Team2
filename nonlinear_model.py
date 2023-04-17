# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 20:41:20 2023

@author: Vince
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import sympy as sp
from sympy.physics.mechanics import dynamicsymbols
from tqdm import *
# import control as ctl

sp.init_printing(use_unicode=True)


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

costate = dynamicsymbols('lambda1:17')
temp = sp.zeros(16,1)
for i in range(16):
    temp[i] = costate[i]
costate = temp

W1 = sp.symbols('W1_1:9')
W2 = sp.symbols('W2_1:9')
R  = sp.symbols('R_1:5')

temp = sp.zeros(8)
for i in range(8):
    temp[i,i] = W1[i]
W1 = temp
temp = sp.zeros(8)
for i in range(8):
    temp[i,i] = W2[i]
W2 = temp
temp = sp.zeros(4)
for i in range(4):
    temp[i,i] = R[i]
R = temp

H = 0.5*X1.T*W1*X1+0.5*X2.T*W2*X2+0.5*u.T*R*u+costate.T*Xdot
h = 0.5*X1.T*W1*X1+0.5*X2.T*W2*X2+0.5*u.T*R*u

#conditions
X_dot = sp.zeros(16,1)

for i in range(16):
    X_dot[i] = sp.diff(H,costate[i])

lambda_dot = sp.zeros(16,1)

temp_sym = sp.symbols('temp')
for i in trange(16):
    if i < 8:
        lambda_dot[i] = -(sp.diff(H.subs(X[i+8],temp_sym),X[i])).subs(temp_sym,X[i+8])
    else:
        lambda_dot[i] = -sp.diff(H,X[i])

control_eqs = sp.zeros(4,1)

for i in range(4):
    control_eqs[i] = sp.diff(H,u[i])

mat0,b0 = sp.linear_eq_to_matrix(control_eqs,[u[0],u[1],u[2],u[3]])

controls = mat0**-1*b0

for i in trange(16):
    for j in range(4):
        lambda_dot[i] = lambda_dot[i].subs(u[j],controls[j])
        X_dot[i] = X_dot[i].subs(u[j],controls[j])

params = [m,l,M,g,I_psi,I_theta,I_phi,I_p]
vals = [.3,.242,.56,9.81,.0021,.006178,.006178,.56*.242**2]

w1 = np.array([10,10,10,1,1,1,10,.1])
w2 = np.array([1,1,1,1,1,1,10,.1])
r_ctl = np.array([1,1,1,1])

with tqdm(total = 16*8*2) as pbar:
    for i in range(16):
        for j in range(len(vals)):
            lambda_dot[i] = lambda_dot[i].subs(params[j],vals[j])
            pbar.update(1)
            X_dot[i] = X_dot[i].subs(params[j],vals[j])
            pbar.update(1)

with tqdm(total = 4*16*8+8+4) as pbar:
    for i in range(16):
        for j in range(8):
            lambda_dot[i] = (lambda_dot[i].subs(W1[j,j],w1[j])).subs(W2[j,j],w2[j])
            pbar.update(1)
            X_dot[i] = (X_dot[i].subs(W1[j,j],w1[j])).subs(W2[j,j],w2[j])
            pbar.update(1)
        for j in range(4):
            lambda_dot[i] = lambda_dot[i].subs(R[j,j],r_ctl[j])
            pbar.update(1)
            X_dot[i] = X_dot[i].subs(R[j,j],r_ctl[j])
            pbar.update(1)
    
    for i in range(8):
        h = h.subs(W1[i,i],w1[i]).subs(W2[i,i],w2[i])
        pbar.update(1)
    for i in range(4):
        h = h.subs(R[i,i],r_ctl[i])
        pbar.update(1)
    
x0 = np.array([[-1],
               [-1],
               [-1],
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



##################################################
# Rough Linearization
##################################################
# X_dot = -M_q**-1*((C_q*qdot)+G_q-U)

# cpy = sp.zeros(8,1)
# for i in range(8):
#     cpy[i] = X_dot[i]

# params = [m,l,M,g,I_psi,I_theta,I_phi,I_p]
# vals = [1,1,7,10,1,1,1,1]



# for row in trange(8):
#     for j in range(len(params)):
#         cpy[row] = cpy[row].subs(params[j],vals[j])
        
# # Linearization
# #################################################
# sbls = sp.symbols('x1:17')
# for row in trange(8):
#     for i in range(8):
#         state_var = qdot[i]
#         sub_sym = sbls[i+8]
#         cpy[row] = cpy[row].subs(state_var,sub_sym)
#     for i in range(8):
#         state_var = q[i]
#         sub_sym = sbls[i]
#         cpy[row] = cpy[row].subs(state_var,sub_sym)

# A_ = sp.zeros(16)
# B_ = sp.zeros(16,4)

# with tqdm(total = 8*20) as pbar:
#     for row in range(8):
#         A_[row,row+8] = 1
            
#     for row in range(8):
#         for column in range(16):
#             A_[row+8,column] = sp.diff(cpy[row],sbls[column])
#             pbar.update(1)
#         for column in range(4):
#             B_[row+8,column] = sp.diff(cpy[row],inpts[0])
#             pbar.update(1)
# ########################################################
# #x,y,z,psi,theta,phi,alpha,beta

# def linearize(inQ,inU):
#     tempA = sp.zeros(16)
#     tempB = sp.zeros(16,4)
    
#     for row in range(16):
#         for column in range(16):
#             tempA[row,column] = A_[row,column]
    
#     for row in range(16):
#         for column in range(4):
#             tempB[row,column] = B_[row,column]
    
#     for row in range(8,16):
#         for j in range(16):
#             for column in range(16):
#                 tempA[row,column] = tempA[row,column].subs(sbls[j],inQ[j])
#                 if column < 4:
#                     tempB[row,column] = tempB[row,column].subs(sbls[j],inQ[j])
#         for j in range(4):
#             for column in range(16):
#                 tempA[row,column] = tempA[row,column].subs(inpts[j],inU[j])
#                 if column < 4:
#                     tempB[row,column] = tempB[row,column].subs(inpts[j],inU[j])
    
#     return np.array(tempA).astype(float),np.array(tempB).astype(float)

# #x,y,z,psi,theta,phi,alpha,beta

# eq = np.array([[-1,-1,-1,0,np.pi/7,0,0,0, 0,0,0,0,0,0,0,0]]).T
# initU = np.array([[1,0,0,0]]).T

# A,B = linearize(eq[:,0],initU[:,0])
# C = np.zeros((1,16))
# C[0,0] = 1

# sys = ctl.c2d(ctl.ss(A,B,C,0),.001)

# Ad = sys.A
# Bd = sys.B
# Cd = sys.C
# Dd = sys.D

# P_0 = np.eye(16)

# Q = np.ones(16)
# Q[0] = 1
# Q[2] = 1
# Q[6] = 3
# Q = np.diag(Q)
# R = .01*np.ones(4)

# R = np.diag(R)

# states = [eq]

# P = [P_0]
# F = []

# def relinearize(inQ,inU):
#     A,B = linearize(inQ,inU)
#     C = np.zeros((1,16))
#     C[0,0] = 1
#     sys = ctl.c2d(ctl.ss(A,B,C,0),.001)    
#     return sys.A,sys.B

# N = 10000

# for k in trange(N):
    
#     F_k1 = -np.linalg.inv(R+Bd.T@P[-1]@Bd)@Bd.T@P[-1]@Ad
#     F.append(F_k1)
#     P_k = (Ad+Bd@F[-1]).T@P[-1]@(Ad+Bd@F[-1])+F[-1].T@R@F[-1]+Q
#     P.append(P_k)
    
    
    
# for i in trange(len(F)):
#     states.append(Ad@states[-1]+Bd@F[N-i]@states[-1])
    
# xs = []
# ys = []
# zs = []
# psis = []
# thetas = []
# phis = []
# alphas = []
# betas = []
# xdots = []
# ydots = []
# zdots = []
# psidots = []
# thetadots = []
# phidots = []
# alphadots = []
# betadots = []

# thrust = []
# pitch = []
# yaw = []
# roll = []

# for i in trange(N):
#     thrust.append((F[i]@states[i])[0])
#     pitch.append((F[i]@states[i])[2])
#     yaw.append((F[i]@states[i])[1])
#     roll.append((F[i]@states[i])[3])
    
#     # if i == 1500:
#     #     relinearize(states[-1],)
    
#     xs.append(states[i][0])
#     ys.append(states[i][1])
#     zs.append(states[i][2])
#     psis.append(states[i][3])
#     thetas.append(states[i][4])
#     phis.append(states[i][5])
#     alphas.append(states[i][6])
#     betas.append(states[i][7])
#     xdots.append(states[i][8])
#     ydots.append(states[i][9])
#     zdots.append(states[i][10])
#     psidots.append(states[i][11])
#     thetadots.append(states[i][12])
#     phidots.append(states[i][13])
#     alphadots.append(states[i][14])
#     betadots.append(states[i][15])

# plt.figure(0)
# plt.plot(xs)
# plt.plot(ys)
# plt.plot(zs)
# plt.title('Position')
# plt.legend(['x','y','z'])
# plt.figure(1)
# plt.plot(xdots)
# plt.plot(ydots)
# plt.plot(zdots)
# plt.title('Velocity')
# plt.legend(['x','y','z'])
# plt.grid()
# plt.figure(2)
# plt.plot(psis)
# plt.plot(thetas)
# plt.plot(phis)
# plt.title('Angle')
# plt.legend([r'$\psi$',r'$\theta$',r'$\phi$'])
# plt.grid()
# plt.figure(3)
# plt.plot(psidots)
# plt.plot(thetadots)
# plt.plot(phidots)
# plt.title('Angular Velocity')
# plt.legend([r'$\dot{\psi}$',r'$\dot{\theta}$',r'$\dot{\phi}$'])
# plt.grid()
# plt.figure(4)
# plt.plot(alphas)
# plt.plot(betas)
# plt.title('Swing Angle')
# plt.legend([r'$\alpha$',r'$\beta$'])
# plt.grid()
# plt.figure(5)
# plt.plot(alphadots)
# plt.plot(betadots)
# plt.title('Swing Angular Velocity')
# plt.legend([r'$\dot{\alpha}$',r'$\dot{\beta}$'])
# plt.grid()
# plt.figure(6)
# plt.plot(thrust)
# plt.plot(pitch)
# plt.plot(yaw)
# plt.plot(roll)
# plt.title('Controls')
# plt.legend(['Thrust','pitch','yaw','roll'])
# plt.grid()
