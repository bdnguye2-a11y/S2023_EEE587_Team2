import matplotlib.pyplot as plt
import numpy as np
import math
from math import cos, sin
from scipy.integrate import odeint

cart_pos_x=4

cart_pos_y=1

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
# plt.show()

class LinearSystem():
    def __init__(self) -> None:
        self.A=Ac
        self.B=Bc
        self.C=Cc
        self.D=np.zeros(self.C.shape)
        self.State_ini=x0
        self.State=np.zeros(self.B.shape)
        self.Y=np.array([[0],
                         [0]])
        self.U=0
        self.State_dot=np.zeros(self.State.shape)
        self.dt=0.01


    def simulate_linear_model_zero_input(self, sim_time=15):

        print(self.A.shape)
        
        print(self.B.shape)
        
        print(self.State.shape)
        
        self.U=0.02
        print(self.U)

        time_steps = sim_time/self.dt
        for iter in range(int(time_steps)):    
            self.State_dot=self.A@self.State +self.B*self.U
            self.Y = self.C@self.State
            self.State = self.State + self.State_dot*self.dt
            print(self.Y[1])
            if iter %10==0:
                self.animate_system()

        plt.show()
    def animate_system(self):
        plt.clf()
        # for stopping simulation with the esc key.
        plt.gcf().canvas.mpl_connect(
            'key_release_event',
            lambda event: [exit(0) if event.key == 'escape' else None])
        cart_pos_x=self.Y[0]
        cart_pos_x1=cart_pos_x-0.5
        cart_pos_x2=cart_pos_x+0.5
        pend_x=cart_pos_x+cos(np.pi/2-self.Y[1])
        pend_y=cart_pos_y+sin(np.pi/2-self.Y[1])
        plt.plot([cart_pos_x, pend_x],[cart_pos_y, pend_y],'-b')
        plt.plot([cart_pos_x1, cart_pos_x2], [cart_pos_y, cart_pos_y],'-k')
        plt.plot([cart_pos_x1, cart_pos_x1], [cart_pos_y, cart_pos_y-0.25],'-k')
        plt.plot([cart_pos_x2, cart_pos_x2], [cart_pos_y, cart_pos_y-0.25],'-k')
        # plt.plot(x[iter], y[iter], 'ob')
        plt.ylim(-1,2)

        plt.grid(True)
        plt.pause(0.01)

# Show the animation
# LS=LinearSystem()
# LS.simulate_linear_model_zero_input(15)