import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint


def derive(x, t, u):
    xd = Ac @ x.squeeze() + Bc @ u
    return xd.squeeze()


I = 1
M = 1
m = 1
l = 1
b = 1
g = 1

a22 = -1 * (I + m * l ** 2) * b / (I * (M + m) + M * m * l ** 2)
a23 = (m ** 2 * g * l ** 2) / (I * (M + m) + M * m * l ** 2)
a42 = -1 * m * l * b / (I * (M + m) + M * m * l ** 2)
a43 = m * g * l * (M + m) / (I * (M + m) + M * m * l ** 2)

b2 = I + m * l ** 2 / (I * (M + m) + M * m * l ** 2)
b3 = m * l / (I * (M + m) + M * m * l ** 2)

Ac = np.array([[0, 1, 0, 0],
               [0, a22, a23, 0],
               [0, 0, 0, 1],
               [0, a42, a43, 0]])

Bc = np.array([[0],
               [b2],
               [0],
               [b3]])

Cc = np.array([[1, 0, 0, 0],
               [0, 0, 1, 0]])

x0 = np.array([0, 0, 0, 0])

F = np.array([1])
t = np.linspace(0, 50, 200)

y = Cc @ np.transpose(odeint(derive, x0.squeeze(), t, args=(F,)))

fig = plt.figure()
plt.grid()
plt.plot(t, y[0, :])
plt.title('Position')
plt.show()
