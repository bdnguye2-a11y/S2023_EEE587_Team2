import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from Library.KalmanFilter import KalmanFilter


def pendulumEqn(x_val, t, m, l, b, g):
    x = np.zeros(2)
    x[0] = x_val[1]
    x[1] = -(b / (m * l * l)) * x_val[1] - (g / l) * np.sin(x_val[0])
    return x


def getSimResult(x_init, m, l, b, g, time, samples):
    t = np.linspace(0.0, time, samples)  # time
    y = odeint(pendulumEqn, x_init, t, args=(m, l, b, g))
    return y


def main():
    # ~~~~~~~~~~~~~~~~ Actual Model
    x_init = np.array([np.pi / 18.0, 0.0])
    m = 1  # kg
    l = 0.5  # m
    b = 0.0  # put 0 for no damping
    g = 9.81
    time = 20
    samples = 3001
    y = getSimResult(x_init, m, l, b, g, time, samples)

    # ~~~~~~~~~~~~~~~~ Kalman Filter Verification begins
    measurementNoiseVariance = 0.02
    z = y.copy()
    z[:, 0] += np.random.normal(0, measurementNoiseVariance, samples)
    z[:, 1] = 0

    # Initialization
    x_hat = np.zeros([2, samples])  # A posteri estimate of x
    P = np.ones([2, 2])  # A posteri error estimate

    # Kalman Filter Tuning Terms
    Q = np.array([[0, 0], [0, 1e-3]])  # process variance
    R = np.array([[50 ** 2], [100 ** 2]])  # measurement variance

    # initial state estimates
    x_hat[0][0] = np.pi / 15.0  # system is at np.pi/15 degrees
    x_hat[1][0] = 0.0  # system is at 0 angular velocity

    # Linear model
    A = np.array([[0, 1], [-g / l, -b / (m * l * l)]])
    B = np.zeros(A.size)  # no input
    C = np.array([[1, 1]])
    U = np.zeros(A.size)  # no input

    KFilter = KalmanFilter(A=A, B=B, C=C, dt=time / (samples - 1))
    KFilter.set_variance(Q, R)
    KFilter.set_initial_error(P=P[:, :, 0])

    for i in range(1, samples):
        # Prediction
        KFilter.predict(U=U, x_hat=x_hat[:, i - 1:i])
        # Update
        x_hat[:, i:i + 1] = KFilter.update(y_meas=z[i])

    timer = np.linspace(0, time, samples)

    # Angle plot
    plt.figure(1)
    plt.xlabel("time(s)")
    plt.ylabel("Rad")
    plt.plot(timer, y[:, 0], 'g-', label="Measurement")
    plt.plot(timer, x_hat[0, :], 'b--', label="Kalman Estimate")
    plt.title("Kalman Filter Angle Response")
    plt.legend()

    # Angular Velocity PLot
    plt.figure(2)
    plt.xlabel("time(s)")
    plt.ylabel("Rad/s")
    plt.plot(timer, y[:, 1], 'g-', label="Measurement")
    plt.plot(timer, x_hat[1, :], 'b--', label="Kalman Estimate")
    plt.title("Kalman Filter Angular Velocity Estimate")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
