# -*- coding: utf-8 -*-
"""
Created on Sat Apr 15 22:08:02 2023

@author: Vince
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 26 15:43:28 2023

@author: vince
"""
%matplotlib
#%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sympy as sp
from tqdm import *
import scipy.integrate

#initial drone physical parameters (guessed at for now)
M = 2
m = .5
d = .015
Ipsi = 1
Ithe = 1
Iphi = 1
Ip = .1
l = .1
g = 9.81

C_T = 1.0
C_D = 1.0
rho = 1.293
R = .0763/2

d_prop = C_D*4*rho*R**5/np.pi**3
b = C_T**4*rho*R**4/np.pi**2

T = lambda omega: b*omega**2
D = lambda omega: d_prop*omega**2

#notes for helping to understand structure of q vector
#q = [x,y,z,psi,theta,phi,alpha,beta].T
#     0,1,2, 3 ,  4  , 5 ,  6  ,  7 #all position variables 
#     8,9,10,11, 12  , 13, 14  , 15 #first derivative terms of above

#initial q vector (initial states)
#              x,y,z  ,s,t           ,f,a,b,x,y,z,s,t,f,a ,b - this is to help with defining initial states
q0 = np.array([0,0,2.0,0,0,0,np.pi/4,0,0,0,0,0,0,0,0,.1])
# confirmed that alpha and beta are relative to inertial frame compare the dynamics evolving from the following two initial cases:
#q0 = np.array([0,0,2.0,0,np.pi/2*.999,0,-np.pi/2*.999,0,0,0,0,0,0,0,0,0]) -> no torque on payload swing bar therefore drone is parallel to payload hanging angle
#q0 = np.array([0,0,2.0,0,np.pi/2*.999,0,0,0,0,0,0,0,0,0,0,0]) -> torque on payload swing bar, therefore drone is perpendicular to payload hanging angle

#the following lambda functions constitute the dynamics of the quadrotor based on the paper

#utility lambda functions for simplifying writing out sin and cos a bunch of times
s = lambda angle: np.sin(angle)
c = lambda angle: np.cos(angle)

J = lambda q:np.array([[Ipsi*s(q[4])**2+Ithe*c(q[4])**2*s(q[5])**2+Iphi*c(the)**2*c(q[5])**2,c(q[4])*c(q[5])*s(q[5])*(Ithe-Iphi),-Ipsi*s(q[4])],
                     [c(q[4])*c(q[5])*s(q[5])*(Ithe-Iphi),Ithe*c(q[5])**2+Iphi*s(q[5])**2,0],
                     [-Iphi*s(q[4]),0,Ipsi],
                     ])
#M(q) matrix terms as part of lagrangian
#some terms are replicated, e.g. m11 = m22 = m33- I just went by whatever came first and recycled
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

#M(q) matrix itself
M_q = lambda q: np.array([[m11,0,0,0,0,0,m17(q),m18(q)],
                          [0,m11,0,0,0,0,m27(q),m28(q)],
                          [0,0,m11,0,0,0,m37(q),0],
                          [0,0,0,m44(q),m45(q),-Ipsi*s(q[4]),0,0],
                          [0,0,0,m45(q),m55(q),0,0,0],
                          [0,0,0,-Ipsi*s(q[4]),0,Ipsi,0,0],
                          [m17(q),m27(q),m37(q),0,0,0,m77,0],
                          [m18(q),m28(q),0,0,0,0,0,m88(q)]])

#C(q,q_dot) matrix terms
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

#C(q,q_dot) matrix itself
C_q = lambda q: np.array([[0,0,0,0,0,0,c17(q),c18(q)],
                          [0,0,0,0,0,0,c27(q),c28(q)],
                          [0,0,0,0,0,0,m*l*c(q[6])*q[14],0],
                          [0,0,0,c44(q),c45(q),c46(q),0,0],
                          [0,0,0,c54(q),c55(q),c56(q),0,0],
                          [0,0,0,c64(q),c65(q),0,0,0],
                          [0,0,0,0,0,0,0,-m*l**2*s(q[6])*c(q[6])*q[15]],
                          [0,0,0,0,0,0,m*l**2*s(q[6])*c(q[6])*q[15],m*l**2*s(q[6])*c(q[6])*q[14]]])
#G(q) matrix (easy peasy)
G_q = lambda q: np.array([[0],
                          [0],
                          [(M+m)*g],
                          [0],
                          [0],
                          [0],
                          [m*l*g*s(q[6])],
                          [0]])

#b(q) matrix used in paper to relate control inputs U to system states
b_q = lambda q:np.array([[s(q[6])*s(q[3])+c(q[5])*c(q[3])*s(q[4]),0,0,0],
                         [c(q[5])*s(q[4])*s(q[3])-c(q[3])*s(q[5]),0,0,0],
                         [c(q[4])*c(q[5]),0,0,0],
                         [0,1,0,0],
                         [0,0,1,0],
                         [0,0,0,1],
                         [0,0,0,0],
                         [0,0,0,0]])

#Our Control inputs - here it is being initialized to counteract gravity almost exactly

def set_rotor_speed(inpt,variation=0.1):
    omega1 = inpt*(1+2*(.5-np.random.random())*variation)
    omega2 = inpt*(1+2*(.5-np.random.random())*variation)
    omega3 = inpt*(1+2*(.5-np.random.random())*variation)
    omega4 = inpt*(1+2*(.5-np.random.random())*variation)
    
    temp1 = T(omega1)
    temp2 = T(omega2)
    temp3 = T(omega3)
    temp4 = T(omega4)
    
    temp_vec = np.array([[temp1+temp2+temp3+temp4],
                         [l*(temp4-temp2)],
                         [l*(temp3-temp1)],
                         [-D(omega1)+D(omega1)-D(omega3)+D(omega4)]])
    return temp_vec

#two utility functions needed as part of the scipy solvers
#created two to avoid having to reedit to experiment between the two solvers
def evolve_solve_ivp(innt,inq,inU,noise=0):
    temp = np.zeros(16)
    temp[:8] = inq[8:]
    temp[8:]=(np.linalg.inv(M_q(inq.squeeze()))@(b_q(inq.squeeze())@inU - C_q(inq.squeeze())@inq[8:].reshape(8,1)-G_q(inq.squeeze()))).squeeze()
    return temp

def evolve_odeint(inq,innt,inU,noise=0):
    temp = np.zeros(16)
    temp[:8] = inq[8:]
    temp[8:]=(np.linalg.inv(M_q(inq.squeeze()))@(b_q(inq.squeeze())@inU - C_q(inq.squeeze())@inq[8:].reshape(8,1)-G_q(inq.squeeze()))).squeeze()
    return temp

#number of time divisions within our linear space for simulation
#with scipy solve_ivp the solver handles discretizing the time interval itself
#so this ends up being a minimum number of time steps there
#likely most analagous to sampling time if we wanted to move to discrete time
stps = 1000
time = 10
t = np.linspace(0,time,stps)

#initializing our state vector
q = q0.copy()

#initializing an output vector for plottiing results
y = np.array([q])

#model simulation loop- this is where we split the time interval so that we can modulate the control signal
controls = []
for i in trange(stps-1):
    U = set_rotor_speed(4700,0.01)
    controls.append(U.T)
    interval = (t[i],t[i+1])
#    q = scipy.integrate.odeint(evolve_odeint,q,interval,args=(U,t))[-1]
#    y = np.concatenate((y,q.reshape(1,16)),0)
    output = scipy.integrate.solve_ivp(evolve_solve_ivp,interval,q,method='LSODA',args=(U,0))
    q = output.y[:,-1]
    y = np.concatenate((y,output.y.T),0)

controls = np.array(controls).squeeze()

#alternate simulation scheme (without control modulation) mostly used here for investigating different solver accuracies
#y = scipy.integrate.odeint(evolve_odeint,q,t,args=(U,t))
#output = scipy.integrate.solve_ivp(evolve_solve_ivp,(t[0],t[-1]),q,method='BDF',t_eval=t,args=(U,0))
#y = output.y.T

#notes
#RK/Radau/BOP853 Methods seem to lack accuracy
#BDF seems slowest but most accurate- 
#LSODA seems like a good comprimise between accuracy and speed

#Update unsure if swing attenuation is correct behavior or any product of solver error

#reference: https://danielmuellerkomorowska.com/2021/02/16/differential-equations-with-scipy-odeint-or-solve_ivp/
#scipy.integrate.odeint(evolve,q,t,args=(U,t))


#calculate payload position from state angles (for visualization)
payload_position = np.zeros((y.shape[0],3))
init = np.array([[0],[0],[-l]])
for i in range(y.shape[0]):
    payload_position[i,:] = y[i,:3]+np.array([l*np.cos(y[i,7])*np.sin(y[i,6]),l*np.sin(y[i,6])*np.sin(y[i,7]),-l*np.cos(y[i,6])])

#Let's animate this for better effect
#example taken from https://matplotlib.org/stable/gallery/animation/random_walk.html
fig_anim = plt.figure()
plt.title('Animation of Quadrotor and Payload Positions')
plt.legend(['Quadrotor','Payload'])
ax_anim = fig_anim.add_subplot(projection='3d')
lines = [ax_anim.plot([],[],[])[0],ax_anim.plot([],[],[])[0]]
# Setting the axes properties
xlims = (min(y[:,0].min(),payload_position[:,0].min()), max(y[:,0].max(),payload_position[:,0].max()))
if xlims[0] == xlims[1]:
    if xlims[0] == 0:
        xlims = (0,1)
    else:
        xlims = (0,xlims[0])
ylims = (min(y[:,1].min(),payload_position[:,1].min()), max(y[:,1].max(),payload_position[:,1].max()))
if ylims[0] == ylims[1]:
    if ylims[0] == 0:
        ylims = (0,1)
    else:
        ylims = (0,ylims[0])
zlims = (min(y[:,2].min(),payload_position[:,2].min()), max(y[:,2].max(),payload_position[:,2].max()))
if zlims[0] == zlims[1]:
    if zlims[0] == 0:
        zlims = (0,1)
    else:
        zlims = (0,zlims[0])

lims = (min(xlims[0],ylims[0],zlims[0]),max(xlims[0],ylims[0],zlims[0]))
ax_anim.set(xlim3d=(lims[0],lims[1]), xlabel='X')
ax_anim.set(ylim3d=(lims[0],lims[1]), ylabel='Y')
ax_anim.set(zlim3d=(lims[0],lims[1]), zlabel='Z')

#lims = (min(xlims,ylims,zlims),max(xlims,ylims,zlims))

#ax_anim.set(xlim3d=(lims[0],lims[1]), xlabel='X')
#ax_anim.set(ylim3d=(lims[0],lims[1]), ylabel='Y')
#ax_anim.set(zlim3d=(lims[0],lims[1]), zlabel='Z')

walks = [y[:,:3],payload_position]
def animate(num,lines,walks):
    for line,walk in zip(lines,walks):
        line.set_data(walk[:num, :2].T)
        line.set_3d_properties(walk[:num, 2])


ani = animation.FuncAnimation(fig_anim,animate,fargs=(lines,walks),interval=10)

#plotting 3d positions of drone and payload
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.plot(y[:,0],y[:,1],y[:,2])
ax.plot(payload_position[:,0],payload_position[:,1],payload_position[:,2])
plt.title('Quadrotor and Payload position')
plt.legend(['Quadrotor','Payload'])
# histdd = np.array(histdd).squeeze()

#some plots for better understanding drone behaviour
fig = plt.figure()
plt.grid()
plt.plot(y[:,0])
plt.plot(y[:,1])
plt.plot(y[:,2])
plt.legend(['X','Y','Z'])
plt.title('Position')

fig = plt.figure()
plt.grid()
plt.plot(y[:,8])
plt.plot(y[:,9])
plt.plot(y[:,10])
plt.legend(['x-dot','y-dot','z-dot'])
plt.title('Velocity')

fig = plt.figure()
plt.grid()
plt.plot(y[:,3])
plt.plot(y[:,4])
plt.plot(y[:,5])
plt.legend(['psi','theta','phi'])
plt.title('Angle')

fig = plt.figure()
plt.grid()
plt.plot(y[:,6])
plt.plot(y[:,7])
plt.legend(['alpha','beta'])
plt.title('Payload Angles')

fig = plt.figure()
plt.grid()
plt.plot(payload_position[:,0])
plt.plot(payload_position[:,1])
plt.plot(payload_position[:,2])
plt.legend(['X','Y','Z'])
plt.title('Payload Position')
plt.show()

fig = plt.figure()
plt.grid()
plt.plot(controls[:,0])
plt.plot(controls[:,1])
plt.plot(controls[:,2])
plt.plot(controls[:,3])
plt.legend(['U1','U2','U3','U4'])
plt.title('Control Signals')
plt.show()

ani.repeat = True