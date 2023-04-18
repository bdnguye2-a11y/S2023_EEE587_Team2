import numpy as np


class KalmanFilter:
    def __init__(self, A, B, C, dt):
        self.A_hat = np.identity(A.shape[0]) + A * dt
        self.B_hat = B * dt
        self.C = C

        self.x_hat_bar = None
        self.P_bar = None
        self.P = np.ones((self.A_hat.shape[0], self.A_hat.shape[0]))

        self.Q = None
        self.R = None

    def set_variance(self, Q, R):
        self.Q = Q
        self.R = R

    def set_initial_error(self, P):
        self.P = P

    def predict(self, U, x_hat):
        self.x_hat_bar = self.A_hat @ x_hat + self.B_hat @ U  # need to add B@U
        self.P_bar = self.A_hat @ self.P[:, :] @ self.A_hat.transpose() + self.Q

    def update(self, y_meas):
        K = np.divide(self.P_bar[:, :] @ self.C.transpose(), self.C @ self.P_bar[:, :] @ self.C.transpose() + self.R)
        self.P = self.P_bar[:, :] - K @ self.C @ self.P_bar[:, :]
        x_hat = self.x_hat_bar[:] + K * (y_meas[:].reshape(K.shape) - self.x_hat_bar[:])

        return x_hat
