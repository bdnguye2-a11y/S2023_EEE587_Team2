import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint


def derive(x, t, u):
    xd = Ac @ x.squeeze() + np.transpose(Bc) @ u
    return xd.squeeze()


l = 0.15
m = 0.3
M = 1
Ip = 0.1
Izz = 1
Iyy = 1
Ixx = 1
g = 9.81

a2_9 = g * (2 * l ** 2 * m ** 2 + M * l ** 2 * m + Ip * m + Ip * M) / ((M + m) * (m * l ** 2 + Ip))
a2_13 = g * l ** 2 * m ** 2 * (2 * l ** 2 * m ** 2 + M * l ** 2 * m + Ip * m + Ip * M)
a4_11 = -g * (2 * l ** 2 * m ** 2 + M * l ** 2 * m + Ip * m + Ip * M) / ((M + m) * (m * l ** 2 + Ip))
a4_15 = g * l ** 2 * m ** 2 * (2 * l ** 2 * m ** 2 + M * l ** 2 * m + Ip * m + Ip * M)
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

Bc = np.array([[0, 0, 0, 0, 0, 1 / (M + m), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 1 / Izz, 0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0, 1 / Iyy, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 1 / Ixx, 0, 0, 0, 0, 0, 0]])

Cc = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]])

x0 = np.array([0, 0, 0, 0, 0, 2.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

F = np.array([[(M+m)*g],
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