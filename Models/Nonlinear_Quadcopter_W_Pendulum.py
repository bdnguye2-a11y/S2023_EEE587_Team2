import numpy as np

class NLQUADCOPTER_PAYLOAD:
    def __init__(self):
        self.M = .5
        self.m = 2
        self.d = .015
        self.Ipsi = 1
        self.Ithe = 1
        self.Iphi = 1
        self.Ip = .1
        self.l = 10
        self.g = 9.81

        self.s = lambda angle: np.sin(angle)
        self.c = lambda angle: np.cos(angle)

    def EulerLangrange(self, q, num):

        M_q = self.InertiaMat(q)
        C_q = self.CentriMat(q)
        G_q = self.GravityMat(q)
        b_q = self.ControlMatrix(q)
        U = self.InputControl(num)

        temp = np.zeros(16)
        temp[:8] = q[8:]
        temp[8:] = (np.linalg.inv(M_q)@ (b_q @ U - C_q @ q[8:].reshape(8, 1) - G_q)).squeeze()
        return temp

    def InertiaMat(self, q):
        M = self.M
        m = self.m
        Ipsi = self.Ipsi
        Ithe = self.Ithe
        Iphi = self.Iphi
        Ip = self.Ip
        l = self.l

        c = self.c
        s = self.s

        m11 = M + m
        m17 = m * l * c(q[6]) * c(q[7])
        m18 = -m * l * s(q[6]) * s(q[7])
        m27 = m * l * c(q[6]) * s(q[7])
        m28 = m * l * s(q[6]) * c(q[7])
        m37 = m * l * s(q[6])
        m44 = Ipsi * s(q[4]) ** 2 + c(q[4]) ** 2 * (Ithe * s(q[5]) ** 2 + Iphi * c(q[5]) ** 2)
        m45 = (Ithe - Iphi) * (c(q[4]) * s(q[5]) * c(q[5]))
        m55 = Ithe * c(q[5]) ** 2 + Iphi * s(q[5]) ** 2
        m77 = m * l ** 2 + Ip
        m88 = m * l ** 2 * s(q[6]) ** 2 + Ip

        M_q = np.array([[m11, 0, 0, 0, 0, 0, m17, m18],
                                  [0, m11, 0, 0, 0, 0, m27, m28],
                                  [0, 0, m11, 0, 0, 0, m37, 0],
                                  [0, 0, 0, m44, m45, -Ipsi * s(q[4]), 0, 0],
                                  [0, 0, 0, m45, m55, 0, 0, 0],
                                  [0, 0, 0, -Ipsi * s(q[4]), 0, Ipsi, 0, 0],
                                  [m17, m27, m37, 0, 0, 0, m77, 0],
                                  [m18, m28, 0, 0, 0, 0, 0, m88]])

        return M_q

    def CentriMat(self, q):
        m = self.m
        Ipsi = self.Ipsi
        Ithe = self.Ithe
        Iphi = self.Iphi
        l = self.l
        c = self.c
        s = self.s

        c17 = -m*l*(c(q[6])*s(q[7])*q[15]+s(q[6])*c(q[7])*q[14])
        c18 = -m * l * (c(q[6]) * s(q[7]) * q[14] + s(q[6]) * c(q[7]) * q[15])
        c27 = m * l * (c(q[6]) * c(q[7]) * q[15] - s(q[6]) * s(q[7]) * q[14])
        c28 = m * l * (c(q[6]) * c(q[7]) * q[14] - s(q[6]) * s(q[7]) * q[15])
        c44 = Ipsi * q[12] * s(q[4]) * c(q[4]) - (Ithe + Iphi) * (q[12] * s(q[4]) * c(q[4]) * s(q[5]) ** 2) + \
                        (Ithe - Iphi) * q[13] * c(q[4]) ** 2 * s(q[5]) * c(q[5])
        c45 = Ipsi * q[11] * s(q[4]) * c(q[4]) - (Ithe - Iphi) * (
                    q[12] * s(q[4]) * c(q[5]) * s(q[5]) + q[13] * c(q[4]) * s(q[5]) ** 2) \
                        - (Ithe + Iphi) * (q[11] * s(q[4]) * c(q[4]) * c(q[5]) ** 2 - q[13] * c(q[4]) * c(q[5]) ** 2)

        c46 = -(Ipsi * q[12] * c(q[4]) - (Ithe - Iphi) * (q[11] * c(q[4]) ** 2 * s(q[5]) * c(q[5])))
        c54 = q[11] * s(q[4]) * c(q[4]) * (-Ipsi + Ithe * s(q[5]) ** 2 + Iphi * c(q[5]) ** 2)
        c55 =  -(Ithe - Iphi) * (q[13] * s(q[5]) * c(q[5]))
        c56 = Ipsi * q[11] * c(q[4]) + (Ithe - Iphi) * (
                    -q[12] * s(q[4]) * c(q[5]) + q[11] * c(q[4]) * c(q[5]) ** 2 - q[11] * c(q[4]) * s(q[5]) ** 2)
        c64 = -(Ithe - Iphi) * (q[11] * c(q[4]) ** 2 * s(q[5]) * c(q[5]))
        c65 = -Ipsi * q[11] * c(q[4]) + (Ithe - Iphi) * (
                    q[12] * s(q[5]) * c(q[5]) + q[11] * c(q[4]) * s(q[5]) ** 2 - q[11] * c(q[4]) * c(q[5]) ** 2)

        C_q = np.array([[0, 0, 0, 0, 0, 0, c17, c18],
                                  [0, 0, 0, 0, 0, 0, c27, c28],
                                  [0, 0, 0, 0, 0, 0, m * l * c(q[6]) * q[14], 0],
                                  [0, 0, 0, c44, c45, c46, 0, 0],
                                  [0, 0, 0, c54, c55, c56, 0, 0],
                                  [0, 0, 0, c64, c65, 0, 0, 0],
                                  [0, 0, 0, 0, 0, 0, 0, -m * l ** 2 * s(q[6]) * c(q[6]) * q[15]],
                                  [0, 0, 0, 0, 0, 0, m * l ** 2 * s(q[6]) * c(q[6]) * q[15],
                                   m * l ** 2 * s(q[6]) * c(q[6]) * q[14]]])

        return C_q

    def ControlMatrix(self, q):
        c = self.c
        s = self.s
        b_q = np.array([[s(q[6]) * s(q[3]) + c(q[5]) * c(q[3]) * s(q[4]), 0, 0, 0],
                                  [c(q[5]) * s(q[4]) * s(q[3]) - c(q[3]) * s(q[5]), 0, 0, 0],
                                  [c(q[4]) * c(q[5]), 0, 0, 0],
                                  [0, 1, 0, 0],
                                  [0, 0, 1, 0],
                                  [0, 0, 0, 1],
                                  [0, 0, 0, 0],
                                  [0, 0, 0, 0]])

        return b_q

    def GravityMat(self, q):
        M = self.M
        m = self.m
        g = self.g
        l = self.l

        s = self.s
        G_q = np.array([[0],
                                  [0],
                                  [(M + m) * g],
                                  [0],
                                  [0],
                                  [0],
                                  [m * l * g * s(q[6])],
                                  [0]])

        return G_q

    def InputControl(self, num):
        return num
