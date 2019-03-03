import numpy as np
import scipy.constants as const
import matplotlib.pyplot as plt
import sys
plt.style.use("ggplot")

class SolarCore:

    def __init__(self):
        ## Constants
        self.sigma = 5.67e-8#const.Stefan_Boltzmann
        self.k = const.Boltzmann
        self.m_u = const.atomic_mass
        self.G = const.gravitational_constant
        self.NA = const.Avogadro

        # Mass Fractions
        X = 0.7
        Y3 = 1e-10
        Y = 0.29
        Z = 0.01
        Z_Li = 1e-13
        Z_Be = 1e-13

        self.L0 = 2*3.846e26  # [W]
        self.R0 = 0.72 * 6.96e8  # [m]
        self.M0 = 0.8 * 1.989e30  # [kg]
        self.T0 = 5.7e6  # [K]
        self.rho0 = 5.1 * 1.62e5  # [kg/m^3]
        # molecular average weight
        self.mu = 1 / (2 * X + 3 / 4 * Y + 9 / 14 * Z)
        self.end_step = 0


        self.P0 = self.P(self.rho0, self.T0)

    def opacity(self, T, rho):
        """
        Reads T, rho and kappa values from opacity.txt, returns best fit for kappa.
        :param T: temperature
        :param rho: density
        :return: kappa
        """
        with open("opacity.txt", "r") as infile:
            log_R = np.asarray(infile.readline().split()[1:], dtype=np.float64)

            infile.readline() #empty line

            log_T = []
            log_k = []

            for line in infile:
                log_T.append(line.split()[0]) #first column is log10(T)
                log_k.append(line.split()[1:]) #rest is log10(kappa)

        # actual R value
        R = rho/(T*1e-6)

        # find best fit
        delta_T = np.abs(10**(np.asarray(log_T, dtype=np.float64)) - T)
        delta_R = np.abs(10**(log_R) - R)
        x = np.argmin(delta_T)
        y = np.argmin(delta_R)

        kappa = 10**(np.asarray(log_k, dtype=np.float64)[x, y])

        return kappa

    def rho(self, P, T): #From P = P_G + P_rad
        rho = self.mu * self.m_u / (self.k * T) * (P - ((4 * self.sigma / const.speed_of_light) / 3) * T**4) # P_G + P_R
        return rho

    def P(self, rho, T):
        P = rho*self.k*T/(self.mu*self.m_u) + ((4 * self.sigma / const.speed_of_light) / 3) * T**4
        return P

    def energy_production(self, T, rho):

        ## mass distribution
        X = 0.7
        Y_3 = 1e-10
        Y = 0.29
        Z = 0.01
        Z_Li = 1e-13
        Z_Be = 1e-13

        ## Number densities
        n_x = X * rho / self.m_u
        n_y3 = Y_3 * rho / (3 * self.m_u)
        n_y = Y * rho / (4 * self.m_u)
        n_zBe = Z_Be * rho / (7 * self.m_u)
        n_zLi = Z_Li * rho / (7 * self.m_u)
        n_z = n_zBe + n_zLi
        n_e = n_x + 2 * n_y + n_z
        #print("Oke?", n_e)

        ## lambdas
        T9 = T/1e9
        cm3ps_to_m3ps = 1e-6

        l_pp = ((4.01e-15 * T9**(-2/3) * np.exp(-3.380 * T9**(-1/3)) * (1 + 0.123 * T9**(1/3) + 1.09 * T9**(2/3) + 0.938 * T9))/self.NA)*cm3ps_to_m3ps

        l_33 = ((6.04e10 * T9**(-2/3) * np.exp(-12.276 * T9**(-1/3)) * (1 + 0.034 * T9**(1/3) - 0.522 * T9**(2/3) - 0.124 * T9 + 0.353 * T9**(4/3) + 0.213 * T9**(-5/3)))/self.NA)*cm3ps_to_m3ps

        T9_temp = T9/(1 + 4.95e-2 * T9)
        l_34 = ((5.61e6 * T9_temp**(5/6) * T9**(-3/2) * np.exp(-12.826 * T9_temp**(-1/3))) / self.NA)*cm3ps_to_m3ps

        # check upper limit for Be
        if (T < 1e6):
            l_e7 = (1.57e-7 / (n_e * self.NA))*cm3ps_to_m3ps
        else:
            l_e7 = ((1.34e-10 * T9**(-1/2) * (1 - 0.537 * T9**(1/3) + 3.86 * T9**(2/3) + 0.0027 * T9**(-1) * np.exp(2.515e-3 * T9**(-1)))) / self.NA)*cm3ps_to_m3ps

        T9_temp = T9/(1 + 0.759 * T9)
        l_17Li = ((1.096e9 * T9**(-2/3) * np.exp(-8.472 * T9**(-1/3)) - 4.830e8 * T9_temp**(5/6) * T9**(-3/2) * np.exp(-8.472 * T9_temp**(-1/3)) + 1.06e10 * T9**(-3/2) * np.exp(-30.442 * T9**(-1))) / self.NA)*cm3ps_to_m3ps
        l_17Be = ((3.11e5 * T9**(-2/3) * np.exp(-10.262 * T9**(-1/3)) + 2.53e3 * T9**(-3/2) * np.exp(-7.306 * T9**(-1))) / self.NA)*cm3ps_to_m3ps

        ## rates
        r_pp = l_pp * (n_x**2) / (rho * 2)
        r_33 = l_33 * (n_y3**2) / (rho * 2)
        r_34 = l_34 * (n_y3 * n_y) / rho
        r_e7 = l_e7 * (n_zBe * n_e) / rho
        r_17Li = l_17Li * (n_x * n_zLi) / rho
        r_17Be = l_17Be * (n_x * n_zBe) / rho
        #print("kslale", l_pp, l_33, r_34, l_e7, l_17Li, l_17Be)
        #print("pelg", n_x, n_y3, n_y, n_e, n_zLi, n_zBe)


        ## Energy values
        MeVtoJ = 1.602e-13
        # first two steps, merged to one
        Q_pp = ((0.15 + 1.02) + 5.49)*MeVtoJ  # [J]

        # PP I
        Q_33 = (12.86)*MeVtoJ

        # PP II
        Q_34 = (1.59)*MeVtoJ


        Q_e7 = (0.05)*MeVtoJ


        Q_17Li = (17.35)*MeVtoJ

        # PP III
        Q_17Be = (0.14 + (1.02 + 6.88) + 3.00)*MeVtoJ

        eps = Q_pp * r_pp + Q_33 * r_33 \
                  + Q_34 * r_34 + Q_e7 * r_e7 + Q_17Li * r_17Li \
                  + Q_17Be * r_17Be # + Q_33 * r_33
        #print(Q_pp * r_pp * rho, Q_33 * r_33 * rho, Q_34 * r_34 * rho, Q_e7 * r_e7 * rho, Q_17Li * r_17Li * rho, Q_17Be * r_17Be * rho)
        return eps


    def drdm(self, r, rho):
        return (1/(4 * np.pi * r**2 * rho))

    def dPdm(self, r, m):
        return (- self.G * m / (4 * np.pi * r ** 4))

    def dTdm(self, r, T, L, rho):
        return (-3 * self.opacity(T, rho) * L / (256 * np.pi ** 2 * self.sigma * r ** 4 * T ** 3))

    def euler(self):
        n = int(1e4)
        p_max = 0.005
        r, P, L, T, rho, epsilon, m = np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)

        r[0], P[0], L[0], T[0], rho[0], epsilon[0], m[0] = self.R0, self.P0, self.L0, self.T0, self.rho0, self.energy_production(self.T0, self.rho0), self.M0
        #print(r[0], P[0], L[0], T[0], rho[0], epsilon[0], m[0])
        for i in range(n-1):

            if m[i] < 0 or r[i] < 0 or L[i] < 0 or rho[i] < 0 or P[i] < 0 or T[i] < 0:
                print("Error, value below zero.")
                print("Step:", i)
                print("Mass:", m[i])
                print("Radius:", r[i])
                print("Luminosity:", L[i])
                print("Density:", rho[i])
                print("Pressure:", P[i])
                print("Temperature:", T[i])
                self.end_step = i
                break
            else:
                #if(dynamic_steps):
                #   dm = np.min([np.abs(p_max * r[i] / self.drdm(r[i], rho[i])), np.abs(p_max * P[i] / self.dPdm(r[i], m[i])), np.abs(p_max * r[i] / self.energy_production(T[i], rho[i])), np.abs(p_max * r[i] / self.dTdm(r[i], T[i], L[i], rho[i]))])

                dm = 1.989e30 * 1e-4

                r[i+1] = r[i] - self.drdm(r[i], rho[i]) * dm
                P[i+1] = P[i] - self.dPdm(r[i], m[i]) * dm
                L[i+1] = L[i] - self.energy_production(T[i], rho[i]) * dm
                T[i+1] = T[i] - self.dTdm(r[i], T[i], L[i], rho[i]) * dm

                rho[i+1] = self.rho(P[i], T[i])
                epsilon[i+1] = self.energy_production(T[i], rho[i])
                m[i+1] = m[i] - dm

        return r, P, L, T, rho, epsilon, m

    def plot(self, r, P, L, T, rho, epsilon, m):

        #fig = plt.figure()

        #ax = fig.add_subplot(2, 2, 1)
        plt.title("Radius")
        plt.xlabel("$\\frac{m}{m_{\\mathrm{sun}}}$")
        plt.ylabel("$\\frac{r}{R_{\\mathrm{sun}}}$")
        plt.plot(m[0 : self.end_step]/1.989e30, r[0 : self.end_step]/6.96e8)
        plt.tight_layout()
        plt.show()

        plt.title("Luminosity")
        plt.xlabel("$\\frac{m}{M_{\\mathrm{sun}}}$")
        plt.ylabel("$\\frac{L}{L_{\\mathrm{sun}}}$")
        plt.plot(m[0 : self.end_step] / 1.989e30, L[0 : self.end_step] / 3.846e26)
        plt.tight_layout()
        plt.show()

        plt.title("Temperature")
        plt.xlabel("$\\frac{m}{M_{\\mathrm{sun}}}$")
        plt.ylabel("T[MK]")
        plt.plot(m[0 : self.end_step] / 1.989e30, T[0 : self.end_step]*1e-6)
        plt.tight_layout()
        plt.show()

        plt.title("Density")
        plt.xlabel("$\\frac{m}{M_{\\mathrm{sun}}}$")
        plt.ylabel("$\\frac{\\rho}{\\rho_{\\mathrm{sun}}}$")
        plt.plot(m[0 : self.end_step] / 1.989e30, rho[0 : self.end_step]/ 1.62e5)
        plt.tight_layout()
        plt.show()

        """
        plt.set_title = ("Pressure")
        plt.xlabel("$\\frac{m}{M_{\\mathrm{sun}}}$}}$")
        plt.ylabel("$\\frac{P}{P_{\\mathrm{sun}}}$}}$")
        plt.plot(m / 1.989e30, r / 6.96e8)
        plt.tight_layout()
        plt.show()
        """






f = SolarCore()
#print(f.opacity(5.7e6, 5.1 * 1.62e5))
#print(f.energy_production(10**8, 1.62e5))
r, P, L, T, rho, epsilon, m = f.euler()
f.plot(r, P, L, T, rho, epsilon, m)
#print("e", f.energy_production(1.57e7, 1.62e5))
#print("e", f.energy_production(10**8, 1.62e5))
#print("e", f.energyGeneration(10**8, 1.62e5))
#solver = RungeKutta4(f, [f.R0, f.P0, f.L0, f.T0])
#n = 1e5
#u, m = solver.solve(np.arange(0, n, 1))
#print(u)