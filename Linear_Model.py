import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import sympy as sp
from sympy.physics.mechanics import dynamicsymbols

sp.init_printing(use_unicode=True)

def derive(x, t, u):
    xd = Ac @ x.squeeze() + np.transpose(Bc) @ u
    return xd.squeeze()

def inner(in1,in2):
    retun in1[0]*in2[0]+in1[1]*in2[1]+in1[2]*in2[2]

class vec():
    def __init__(self,ina,inb,inc):
        self.a = ina
        self.b = inb
        self.c = inc
    def diff(self,insym = False):
        if class(insym) == bool:
            if insym == False:
                tempa = sp.diff(self.a)
                tempb = sp.diff(self.b)
                tempc = sp.diff(self.c)
                return vec(tempa,tempb,tempc)
            if class(insym) == vec:
                tempa 

l = 0.15
m = 0.3
M = 1
Ip = 0.1
Izz = 1
Iyy = 1
Ixx = 1
g = 1

###########Vince's Checks################

c = lambda x: sp.cos(x)
s = lambda x: sp.sin(x)
syms = []
# syms.append(sp.symbols('x'))
# syms.append(sp.symbols('xdot'))
# syms.append(sp.symbols('y'))
# syms.append(sp.symbols('ydot'))
# syms.append(sp.symbols('z'))
# syms.append(sp.symbols('zdot'))

# syms.append(sp.symbols('psi'))
# syms.append(sp.symbols('psidot'))
# syms.append(sp.symbols('theta'))
# syms.append(sp.symbols('thetadot'))
# syms.append(sp.symbols('phi'))
# syms.append(sp.symbols('phidot'))

# syms.append(sp.symbols('alpha'))
# syms.append(sp.symbols('alphadot'))
# syms.append(sp.symbols('beta'))
# syms.append(sp.symbols('betadot'))

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


# xddots = sp.symbols('xddot')
# yddots = sp.symbols('yddot')
# zddots = sp.symbols('zddot')
# psiddots = sp.symbols('psiddot')
# thetaddots = sp.symbols('thetaddot')
# phiddots = sp.symbols('phiddot')
# alphaddots = sp.symbols('alphaddot')
# betaddots = sp.symbols('betaddot')

m = sp.symbols('m')
l = sp.symbols('l')
M = sp.symbols('M')
g = sp.symbols('g')

I_x = sp.symbols('I_x')
I_y = sp.symbols('I_y')
I_z = sp.symbols('I_z')

ctrls = []
ctrls.append(sp.symbols('u_1'))
ctrls.append(sp.symbols('tau_psi'))
ctrls.append(sp.symbols('tau_theta'))
ctrls.append(sp.symbols('tau_phi'))

L = 0.5*M

xddot = ctrls[0]*((s(phi)*s(psi)+c(phi)*c(psi)*s(theta))-(m*l*c(alpha)*alphaddots+l*m*s(alpha)*alphadot**2))/(m+M)
yddot = ctrls[0]*((c(phi)*s(theta)*s(psi)-c(psi)*s(phi))-(m*l*c(beta)*betaddots+l*m*s(beta)*betadot**2))/(m+M)
zddot = (ctrls[0]*(c(theta)*c(phi)-(m+M)*g-m*l*(s(alpha)*c(beta)*alphaddots-m*l*(s(beta)*c(alpha)*betaddots-m*l*betadot**2*c(alpha)-m*l*betadot**2*c(beta)+2*M*l*s(beta)*s(alpha)*alphadot*betadot))))/(m+M)

temp1 = thetaddots*I_y*I_z*(c(theta)*s(phi)*c(phi))**2
temp2 = I_x*phiddots*s(theta)
temp3 = I_x*thetadot*s(theta)*c(theta)
temp4 = I_y*(-thetadot*s(theta)*c(theta)*s(phi)**2+phidot*c(theta)**2*s(phi)*c(phi))
temp5 = I_z*(thetadot*s(theta)*c(theta)*c(phi)**2+phidot*s(phi)*c(phi)*c(theta)**2)
temp6 = I_x*psidot*s(theta)*c(theta)
temp7 = thetadot*s(theta)*s(phi)*c(phi)+phidot*c(theta)*s(phi)**2-phidot*c(theta)*c(phi)**2+phidot*s(theta)*c(theta)*s(phi)**2
temp8 = phidot*c(theta)*s(phi)**2-phidot*c(theta)*c(phi)**2-psidot*s(theta)*c(theta)*c(phi)**2+thetadot*s(theta)*s(phi)*c(phi)

psiddot = (ctrls[1]-temp1+temp2-psidot*(temp3+temp4-temp5-thetadot*(temp6 - I_y*(temp7+I_z*(temp8))+phidot*(temp9))/(temp10))

########################################


a2_9 = g * (2 * l ** 2 * m ** 2 + M * l ** 2 * m + Ip * m + Ip * M) / ((M + m) * (m * l ** 2 + Ip))
a2_13 = g * l ** 2 * m ** 2 * (2 * l ** 2 * m ** 2 + M * l ** 2 * m + Ip * m + Ip * M)/((M+m)**2/(m*l**2+Ip)**2)
a4_11 = -g * (2 * l ** 2 * m ** 2 + M * l ** 2 * m + Ip * m + Ip * M) / ((M + m) * (m * l ** 2 + Ip))
a4_15 = g * l ** 2 * m ** 2 * (2 * l ** 2 * m ** 2 + M * l ** 2 * m + Ip * m + Ip * M)/((M+m)**2/(m*l**2+Ip)**2)
a14_9 = -g * l * m * (2 * l ** 2 * m ** 2 + M * l ** 2 * m + Ip * m + Ip * M) / ((M + m) * (m * l ** 2 + Ip) ** 2)
a14_13 = -g * l * m * (2 * l ** 2 * m ** 2 + M * l ** 2 * m + Ip * m + Ip * M) / ((M + m) * (m * l ** 2 + Ip) ** 2)
a16_11 = -g * l * m * (2 * l ** 2 * m ** 2 + M * l ** 2 * m + Ip * m + Ip * M) / ((M + m) * (m * l ** 2 + Ip) ** 2)
a16_15 = -g * l * m * (2 * l ** 2 * m ** 2 + M * l ** 2 * m + Ip * m + Ip * M) / ((M + m) * (m * l ** 2 + Ip) ** 2)

Ac = np.array([[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0, a2_9, 0, 0, 0, a2_13, 0, 0, 0],
               [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, a4_11, 0, 0, 0, a4_15, 0],
               [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0, a14_9, 0, 0, 0, a14_13, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, a16_11, 0, 0, 0, a16_15, 0]])

Bc = np.array([[0, 0, 0, 0, 0, 1 / (M + m), 0, 0,       0, 0,       0, 0,       0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0,           0, 1 / Izz, 0, 0,       0, 0,       0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0,           0, 0,       0, 1 / Iyy, 0, 0,       0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0,           0, 0,       0, 0,       0, 1 / Ixx, 0, 0, 0, 0]])

Cc = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]])

x0 = np.array([0, 0, 0, 0, 2.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

F = np.array([[0],
              [0],
              [0],
              [0]])
t = np.linspace(0, 50, 200)

y = Cc@np.transpose(odeint(derive, x0.squeeze(), t, args=(F.squeeze(),)))
# payload_position = np.zeros((y.shape[0],3))
# init = np.array([[0],[0],[-l]])
# for i in range(y.shape[0]):
#     payload_position[i,:] = y[i,:3]+np.array([l*np.cos(y[i,7])*np.sin(y[i,6]),l*np.sin(y[i,6])*np.sin(y[i,7]),-l*np.cos(y[i,6])])

# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')
# ax.plot(y[:,0],y[:,1],y[:,2])
# ax.plot(payload_position[:,0],payload_position[:,1],payload_position[:,2])
# # histdd = np.array(histdd).squeeze()

fig = plt.figure()
plt.grid()
# plt.plot(y[0, :])
# plt.plot(y[1, :])
plt.plot(y[2, :])
plt.title('Position')

# fig = plt.figure()
# plt.grid()
# plt.plot(y[8, :])
# plt.plot(y[9, :])
# plt.plot(y[10, :])
# plt.title('Velocity')
#
# fig = plt.figure()
# plt.grid()
# plt.plot(y[3, :])
# plt.plot(y[4, :])
# plt.plot(y[5, :])
# plt.title('Angle')
#
# fig = plt.figure()
# plt.grid()
# plt.plot(y[6, :])
# plt.plot(y[7, :])
# plt.title('Payload Angles')
#
# fig = plt.figure()
# plt.grid()
# plt.plot(payload_position[0, :])
# plt.plot(payload_position[1, :])
# plt.plot(payload_position[2, :])
# plt.title('Payload Position')
plt.show()