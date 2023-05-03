import numpy as np


class DynamicProgramming:
    def __init__(self, Ad, Bd, Cd):
        self.P = None
        self.F = None
        self.H = None
        self.Q = None
        self.R = None
        self.A = None
        self.B = None

    def set_system(self, Ad, Bd):
        self.A = Ad
        self.B = Bd

    def set_gain(self, Q, R, H):
        self.Q = Q
        self.R = R
        self.H = H

    def calc_riccati(self, num, x0):
        Ad = self.A
        Bd = self.B
        Q = self.Q
        R = self.R
        x = [x0]
        for i in range(num):
            F_calc = -np.linalg.inv(self.R + Bd.T @ self.P[-1] @ Bd) @ Bd.T @ self.P[-1] @ Ad
            self.F.append(F_calc)
            P_calc = (Ad + Bd @ self.F[-1]).T @ self.P[-1] @ (Ad + Bd @ self.F[-1]) + self.F[-1].T @ R @ self.F[-1] + Q
            self.P.append(P_calc)

        return self.F
