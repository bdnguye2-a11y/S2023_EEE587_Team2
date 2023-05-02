# -*- coding: utf-8 -*-
"""
Created on Mon May  1 13:49:54 2023

@author: vbevilacqua
"""

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
# from Library.KalmanFilter import KalmanFilter
import sympy as sp
from sympy.physics.mechanics import dynamicsymbols
from tqdm import *
import pickle
# import control as ctl

# sp.init_printing(use_unicode=True)
sp.init_printing(pretty_print=False)

class model():
    def __init__(self):
        c = lambda inpt: sp.cos(inpt)
        s = lambda inpt: sp.sin(inpt)
        self.syms = []

        self.syms.append(dynamicsymbols('x'))
        self.syms.append(sp.diff(self.syms[-1]))
        self.syms.append(dynamicsymbols('y'))
        self.syms.append(sp.diff(self.syms[-1]))
        self.syms.append(dynamicsymbols('z'))
        self.syms.append(sp.diff(self.syms[-1]))

        self.syms.append(dynamicsymbols('psi'))
        self.syms.append(sp.diff(self.syms[-1]))
        self.syms.append(dynamicsymbols('theta'))
        self.syms.append(sp.diff(self.syms[-1]))
        self.syms.append(dynamicsymbols('phi'))
        self.syms.append(sp.diff(self.syms[-1]))

        self.syms.append(dynamicsymbols('alpha'))
        self.syms.append(sp.diff(self.syms[-1]))
        self.syms.append(dynamicsymbols('beta'))
        self.syms.append(sp.diff(self.syms[-1]))

        self.t = sp.symbols('t')

        self.x = self.syms[0]
        self.xdot = self.syms[1]
        self.y = self.syms[2]
        self.ydot = self.syms[3]
        self.z = self.syms[4]
        self.zdot = self.syms[5]

        self.psi = self.syms[6]
        self.psidot = self.syms[7]
        self.theta = self.syms[8]
        self.thetadot = self.syms[9]
        self.phi = self.syms[10]
        self.phidot = self.syms[11]

        self.alpha = self.syms[12]
        self.alphadot = self.syms[13]
        self.beta = self.syms[14]
        self.betadot = self.syms[15]

        self.q = sp.Matrix([self.x,self.y,self.z,self.psi,self.theta,self.phi,self.alpha,self.beta])

        self.m = sp.symbols('m')
        self.l = sp.symbols('l')
        self.M = sp.symbols('M')
        self.g = sp.symbols('g')

        self.I_psi = sp.symbols('I_psi')
        self.I_theta = sp.symbols('I_theta')
        self.I_phi = sp.symbols('I_phi')

        self.I_p = sp.symbols('I_p')

        self.J = sp.zeros(3,3)

        s_th = s(self.theta)
        c_th = c(self.theta)
        c_ph = c(self.phi)
        s_ph = s(self.phi)

        self.J[0,0] = self.I_psi*s_th**2 + self.I_theta*c_th**2*s_ph**2+self.I_phi*c_th**2*c_ph**2
        self.J[0,1] = c_th*c_ph*s_ph*(self.I_theta-self.I_phi)
        self.J[1,0] = self.J[0,1]
        self.J[1,1] = self.I_theta*c_ph**2+self.I_phi*s_ph**2
        self.J[0,2] = -self.I_psi*s_th
        self.J[2,0] = self.J[0,2]
        self.J[2,2] = self.I_psi

        m11 = self.M+self.m
        m22 = m11
        m33 = m11
        m17 = self.m*self.l*c(self.alpha)*c(self.beta)
        m71 = m17
        m18 = -self.m*self.l*s(self.alpha)*s(self.beta)
        m81 = m18
        m27 = self.m*self.l*c(self.alpha)*s(self.beta)
        m72 = m27
        m28 = self.m*self.l*s(self.alpha)*c(self.beta)
        m82 = m28
        m37 = self.m*self.l*s(self.alpha)
        m73 = m37
        m44 = self.I_psi*s(self.theta)**2+c(self.theta)**2*(self.I_theta*s(self.phi)**2+self.I_phi*c(self.phi)**2)
        m45 = (self.I_theta-self.I_phi)*(c(self.theta)*s(self.phi)*c(self.phi))
        m54 = m45
        m55 = self.I_theta*c(self.phi)**2+self.I_phi*s(self.phi)**2
        m77 = self.m*self.l**2+self.I_p
        m88 = self.m*self.l**2*s(self.alpha)**2+self.I_p

        self.M_q = sp.Matrix([[m11,0,0,0,0,0,m17,m18],
                          [0,m22,0,0,0,0,m27,m28],
                          [0,0,m33,0,0,0,m37,0],
                          [0,0,0,m44,m45,-self.I_psi*s(self.theta),0,0],
                          [0,0,0,m54,m55,0,0,0],
                          [0,0,0,-self.I_psi*s(self.theta),0,self.I_psi,0,0],
                          [m71,m72,m73,0,0,0,m77,0],
                          [m81,m82,0,0,0,0,0,m88]])

        c17 = -self.m*self.l*(c(self.alpha)*s(self.beta)*self.betadot+s(self.alpha)*c(self.beta)*self.alphadot)
        c18 = -self.m*self.l*(c(self.alpha)*s(self.beta)*self.alphadot+s(self.alpha)*c(self.beta)*self.betadot)
        c27 = self.m*self.l*(c(self.alpha)*c(self.beta)*self.betadot-s(self.alpha)*s(self.beta)*self.alphadot)
        c28 = self.m*self.l*(c(self.alpha)*c(self.beta)*self.alphadot-s(self.alpha)*s(self.beta)*self.betadot)
        c44 = self.I_psi*self.thetadot*s(self.theta)*c(self.theta)-(self.I_theta+self.I_phi)*(self.thetadot*s(self.theta)*c(self.theta)*s(self.phi)**2)+(self.I_theta-self.I_phi)*self.phidot*c(self.theta)**2*s(self.phi)*c(self.phi)
        c45 = self.I_psi*self.psidot*s(self.theta)*c(self.theta)-(self.I_theta-self.I_phi)*(self.thetadot*s(self.theta)*c(self.phi)*s(self.phi)+self.phidot*c(self.theta)*s(self.phi)**2)-(self.I_theta+self.I_phi)*(self.psidot*s(self.theta)*c(self.theta)*c(self.phi)**2-self.phidot*c(self.theta)*c(self.phi)**2)
        c46 = -(self.I_psi*self.thetadot*c(self.theta)-(self.I_theta-self.I_phi)*(self.psidot*c(self.theta)**2*s(self.phi)*c(self.phi)))
        c54 = self.psidot*s(self.theta)*c(self.theta)*(-self.I_psi+self.I_theta*s(self.phi)**2+self.I_phi*c(self.phi)**2)
        c55 = -(self.I_theta-self.I_phi)*(self.phidot*s(self.phi)*c(self.phi))
        c56 = self.I_psi*self.psidot*c(self.theta)+(self.I_theta-self.I_phi)*(-self.thetadot*s(self.theta)*c(self.phi)+self.psidot*c(self.theta)*c(self.phi)**2-self.psidot*c(self.theta)*s(self.phi)**2)
        c64 = -(self.I_theta-self.I_phi)*(self.psidot*c(self.theta)**2*s(self.phi)*c(self.phi))
        c65 = -self.I_psi*self.psidot*c(self.theta)+(self.I_theta-self.I_phi)*(self.thetadot*s(self.phi)*c(self.phi)+self.psidot*c(self.theta)*s(self.phi)**2-self.psidot*c(self.theta)*c(self.phi)**2)

        self.C_q = sp.Matrix([[0,0,0,0,0,0,c17,c18],
                          [0,0,0,0,0,0,c27,c28],
                          [0,0,0,0,0,0,self.m*c(self.alpha)*self.alphadot,0],
                          [0,0,0,c44,c45,c46,0,0],
                          [0,0,0,c54,c55,c56,0,0],
                          [0,0,0,c64,c65,0,0,0],
                          [0,0,0,0,0,0,0,-self.m*self.l**2*s(self.alpha)*c(self.alpha)*self.betadot],
                          [0,0,0,0,0,0,self.m*self.l**2*s(self.alpha)*c(self.alpha)*self.betadot,self.m*self.l**2*s(self.alpha)*c(self.alpha)*self.alphadot]])

        self.G_q = sp.Matrix([0,0,(self.M+self.m)*self.g,0,0,0,self.m*self.l*self.g*s(self.alpha),0])

        self.b_u = sp.zeros(8,4)

        self.b_u[0,0] = s(self.alpha)*s(self.psi)+c(self.phi)*c(self.psi)*s(self.theta)
        self.b_u[1,0] = c(self.phi)*s(self.theta)*s(self.psi)-c(self.psi)*s(self.phi)
        self.b_u[2,0] = c(self.theta)*c(self.phi)
        self.b_u[3,1]=1
        self.b_u[4,2]=1
        self.b_u[5,3]=1

        self.qdot = sp.diff(self.q,self.t)
        self.qddot = sp.diff(self.qdot,self.t)

        inpts = sp.symbols('u1:5')

        self.u = sp.Matrix([inpts[0],inpts[1],inpts[2],inpts[3]])
        
        self.U = self.b_u*self.u

        self.X1 = self.q
        self.X2 = self.qdot

        self.X = sp.Matrix([self.X1,self.X2])
        self.F1 = self.X2
        Z = self.M_q**-1
        N = -Z*(self.C_q*self.X2+self.G_q)
        self.F2 = N+Z*self.U

        self.Xdot = sp.Matrix([self.F1,self.F2])
        
    def define_params(self,input_params,input_vals):
        
        self.Xdot_params = self.Xdot.copy()
        
        for i in range(len(input_params)):
            self.Xdot_params = self.Xdot_params.subs(input_params[i],input_vals[i])
            
class quadrotor():
    def __init__(self,initial_state):
        
        self.model = model()
        self.params = [self.model.m,self.model.l,self.model.M,self.model.g,self.model.I_psi,self.model.I_theta,self.model.I_phi,self.model.I_p]
        self.vals = [.3,.242,.56,9.81,.0021,.006178,.006178,.56*.242**2]
        
        self.model.define_params(self.params,self.vals)
        
        self.state = initial_state
        
        self.a = sp.lambdify((*self.model.X,*self.model.u),self.model.Xdot_params)
        
    def deriv(self,inx,time,inu,tin2=0):
        
        return self.a(*inx,*inu).reshape(16)
    
    def simulate(self,input_function,dT=.001):
        end_condition = False
        otpt = np.array([initial_state.copy()])
        t = 0
        while not end_condition:
            inpt = input_function(self.state,t)
            otpt = np.concatenate((otpt, odeint(self.deriv,self.state,args=(inpt))),0)
            if t > 5:
                end_condition = True
        
x0 = np.zeros(16)
u0 = np.zeros(4)

xr = np.random.random(16)
ur = np.random.random(4)/10

rotor = quadrotor(x0)



