import numpy as np
import scipy.constants as const
from astropy import constants as astroconst
from astropy import units as units
import matplotlib.pyplot as plt

import sys
from scipy.interpolate import interp2d
plt.style.use("ggplot")

class SolarCore:

    def __init__(self):
        """

        ## Todo: nabla-plot går ned på slutten
        ## Todo:   OBS   noen funksjoner endret på , nabla og F_R osv


        """
        ## Constants
        self.sigma = 5.67e-8
        self.k = const.Boltzmann
        self.m_u = const.atomic_mass
        self.G = const.gravitational_constant
        self.NA = const.Avogadro
        self.L_sun = astroconst.L_sun.value
        self.R_sun = astroconst.R_sun.value
        self.M_sun = astroconst.M_sun.value

        ## Mass Fractions
        X = 0.7
        Y3 = 1e-10
        Y = 0.29
        Z = 0.01
        Z_Li = 1e-13
        Z_Be = 1e-13

        ## Initial Values
        self.L0 = 1 * self.L_sun # [W]
        self.R0 = 1 * self.R_sun  # [m]
        self.M0 = 1 * self.M_sun  # [kg]
        self.T0 = 1 * 5770  # [K]
        self.rho0 = 1.42e-7 * 1.408e3  # [kg/m^3]

        ## molecular average weight
        self.mu = 1 / (2 * X + 3 / 4 * Y + 9 / 14 * Z)

        self.c_p = 5 / 2 * self.k / (self.mu * self.m_u)

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

        log_T = np.array(log_T).astype(np.float)
        log_k = np.array(log_k)

        inter = interp2d(log_R, log_T, log_k, kind="linear")
        log_R_val = np.log10(rho * 0.001 / (T * 1e-6) ** 3)
        log_T_val = np.log10(T)
        kappa = inter(log_R_val, log_T_val)
        #print(kappa)
        return 10**kappa[0] * 0.1

    def rho(self, P, T): #From P = P_G + P_rad
        rho = self.mu * self.m_u / (self.k * T) * (P - ((4 * self.sigma / const.speed_of_light) / 3) * T**4) # P_G + P_R
        return rho

    def P(self, rho, T):
        P = rho*self.k*T/(self.mu*self.m_u) + ((4 * self.sigma / const.speed_of_light) / 3) * T**4
        return P

    def energy_production(self, T, rho):
        """
        Calculates energy output of the PP-chains
        :param T: Temperature
        :param rho: Density
        :return: epsilon - Energy
        """
        ## mass distribution
        X = 0.7
        Y_3 = 1e-10
        Y = 0.29
        Z = 0.01
        Z_Li = 1e-7
        Z_Be = 1e-7

        ## Number densities
        n_x = X * rho / self.m_u
        n_y3 = Y_3 * rho / (3 * self.m_u)
        n_y = Y * rho / (4 * self.m_u)
        n_zBe = Z_Be * rho / (7 * self.m_u)
        n_zLi = Z_Li * rho / (7 * self.m_u)
        n_z = n_zBe + n_zLi
        n_e = n_x + 2 * n_y + n_z

        ## lambdas
        T9 = T/1e9
        cm3ps_to_m3ps = 1e-6

        l_pp = ((4.01e-15 * T9**(-2/3) * np.exp(-3.380 * T9**(-1/3)) * (1 + 0.123 * T9**(1/3) + 1.09 * T9**(2/3) + 0.938 * T9))/self.NA)*cm3ps_to_m3ps

        l_33 = ((6.04e10 * T9**(-2/3) * np.exp(-12.276 * T9**(-1/3)) * (1 + 0.034 * T9**(1/3) - 0.522 * T9**(2/3) - 0.124 * T9 + 0.353 * T9**(4/3) + 0.213 * T9**(-5/3)))/self.NA)*cm3ps_to_m3ps

        T9_temp = T9/(1 + 4.95e-2 * T9)
        l_34 = ((5.61e6 * T9_temp**(5/6) * T9**(-3/2) * np.exp(-12.826 * T9_temp**(-1/3))) / self.NA)*cm3ps_to_m3ps

        l_e7 = ((1.34e-10 * T9 ** (-1 / 2) * (1 - 0.537 * T9 ** (1 / 3) + 3.86 * T9 ** (2 / 3) + 0.0027 * T9 ** (-1) * np.exp(2.515e-3 * T9 ** (-1)))) / self.NA) * cm3ps_to_m3ps
        # check upper limit for Be
        if (T < 1e6):
            if(l_e7 > (1.57e-7 / (n_e * self.NA))*cm3ps_to_m3ps):
                l_e7 = (1.57e-7 / (n_e * self.NA)) * cm3ps_to_m3ps

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

        # Check that no step consumes more of an element than the previous steps can produce..
        r_33_34 = (r_33 + r_34)
        if r_33_34 > r_pp:
            # If they do, split the ones that are available accordingly
            r_33 = (r_33 / r_33_34) * r_pp
            r_34 = (r_34 / r_33_34) * r_pp

        r_e7_17Be = (r_e7 + r_17Be)
        if r_e7_17Be > r_34:
            r_e7 = (r_e7 / r_e7_17Be) * r_34
            r_17Be = (r_17Be / r_e7_17Be) * r_34

        if r_17Li > r_e7:
            r_17Li = r_e7

        ## Energy values
        MeVtoJ = 1.602e-13
        # first two steps, merged to one
        Q_pp = ((0.15 + 1.02) + 5.49)*MeVtoJ  # [J]

        # PP I
        Q_33 = (12.86)*MeVtoJ

        # PP II
        Q_34 = (1.59)*MeVtoJ


        Q_e7 = (0.05)*MeVtoJ


        Q_17Li = (17.346)*MeVtoJ

        # PP III
        Q_17Be = 0.14*MeVtoJ#(0.14 + (1.02 + 6.88) + 3.00)*MeVtoJ #0.14*MeVtoJ

        eps = Q_pp * r_pp + Q_33 * r_33 \
                  + Q_34 * r_34 + Q_e7 * r_e7 + Q_17Li * r_17Li \
                  + Q_17Be * r_17Be
        #print(f"{Q_pp * r_pp*rho} + {Q_33 * r_33*rho} + {Q_34 * r_34*rho} + {Q_e7 * r_e7*rho} + {Q_17Li * r_17Li*rho} + {Q_17Be * r_17Be*rho}")
        return eps

    def dr(self, r, rho):
        return 1/(4 * np.pi * r**2 * rho)

    def dP(self, r, m):
        return - self.G * m / (4 * np.pi * r ** 4)

    def dT(self, r, T, L, rho):
        return -3 * self.opacity(T, rho) * L / (256 * np.pi ** 2 * self.sigma * r ** 4 * T ** 3)

    def nabla_rad(self, T, P, L, m, kappa):
        #print(f"{P} / {m} * {T}**4")
        return 3*kappa * L * P / (64 * np.pi * self.sigma * self.G * m * T**4)

    def nabla_ad(self, P, T, rho):
        delta = 1
        return P * delta / (T * rho * self.c_p)
    def total_flux(self, L, r):
        return L/(4 * np.pi * r**2)

    def F_R(self, T, kappa, rho, H_P, nabla):
        return 16*self.sigma * T**4 / (3 * kappa * rho * H_P) * nabla#4 * (4 * self.sigma / const.speed_of_light) * const.speed_of_light * self.G * T**4 * m / (3 * kappa * P * r**2) * nabla

    def F_C(self, F, F_R):
        return F - F_R

    def euler(self):
        """
        Integrate using euler,
        return lists of all values
        """
        n = int(1e5)
        r, P, L, T, rho, epsilon, m = np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)

        r[0], P[0], L[0], T[0], rho[0], epsilon[0], m[0] = self.R0, self.P0, self.L0, self.T0, self.rho0, self.energy_production(self.T0, self.rho0), self.M0
        nabla, nabla_rad, nabla_ad, F_C, F_R, F = np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)

        nabla_rad[0], nabla_ad[0] = self.nabla_rad(T[0], P[0], L[0], m[0], self.opacity(T[0], rho[0])), self.nabla_ad(P[0], T[0], rho[0])

        for i in range(n-1):
            sys.stdout.write('Progress: %4.4f %s\r' % (100 - (m[i] / self.M0) * 100, '%'))
            sys.stdout.flush()


            if r[i] < 1e-4 or L[i] < 1e-4 or rho[i] < 1e-4 or P[i] < 1e-4 or T[i] < 1e-4:
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
            elif m[i] < 1e-4:
                print("Reached zero mass.")
                self.end_step = i
                break
            else:
                ## Check instability

                g = self.G * m[i] / r[i] ** 2

                # Pressure scale height
                H_P = P[i] / (g * rho[i])

                nabla_rad[i+1] = self.nabla_rad(T[i], P[i], L[i], m[i], self.opacity(T[i], rho[i]))
                nabla_ad[i+1] = 0.4#self.nabla_ad(P[i], T[i], rho[i])

                #Energy
                U = (64 * self.sigma * T[i]**3) / (3 * self.opacity(T[i], rho[i]) * rho[i]**2 * self.c_p) * np.sqrt(H_P / (g))

                alpha = 1
                #Mixing length
                l_m = alpha * H_P

                K = 4/l_m**2

                #solve to find xi
                roots = np.roots(np.array([1, U/l_m**2, U**2 / l_m**2 * K, -U/l_m**2 * (nabla_rad[i] - nabla_ad[i])]))
                for root in roots:
                    if np.imag(root) == np.min(np.abs(np.imag(roots))):
                        xi = np.real(root)
                        break


                F[i+1] = self.total_flux(L[i], r[i])

                #print(nabla_rad[i], " > ", nabla_ad[i])
                if nabla_rad[i] > nabla_ad[i]:
                    nabla[i+1] = xi ** 2 + U * K * xi + nabla_ad[i]
                    F_R[i+1] = self.F_R(T[i], self.opacity(T[i], rho[i]), rho[i], H_P, nabla[i])
                    #dT = T[i]/P[i] * nabla[i] * self.dP(r[i], m[i])
                    dT = -self.G * m[i] * T[i] * nabla[i] / (4 * np.pi * r[i]**4 * P[i])
                    F_C[i+1] = self.F_C(F[i], F_R[i])
                else:
                    nabla[i+1] = nabla_rad[i+1]
                    F_R[i+1] = F[i+1]
                    dT = self.dT(r[i], T[i], L[i], rho[i])

                dynamic_steps = True
                p = 0.01
                if (dynamic_steps):
                    dm = np.min([np.abs(p * r[i] / self.dr(r[i], rho[i])), np.abs(p * P[i] / self.dP(r[i], m[i])),
                                 np.abs(p * L[i] / self.energy_production(T[i], rho[i])),
                                 np.abs(p * T[i] / self.dT(r[i], T[i], L[i], rho[i])), np.abs(p * T[i] / (T[i] / P[i] * nabla[i] * self.dP(r[i], m[i])))])

                    # print(dm)
                    # print(np.abs(p * L[i] / self.energy_production(T[i], rho[i])))

                r[i+1] = r[i] - self.dr(r[i], rho[i]) * dm
                P[i+1] = P[i] - self.dP(r[i], m[i]) * dm
                L[i+1] = L[i] - self.energy_production(T[i], rho[i]) * dm
                T[i+1] = T[i] - dT * dm

                #print(f"{L[i]} - {self.energy_production(T[i], rho[i])} * {dm} = {L[i] - self.energy_production(T[i], rho[i]) * dm}")

                #print(self.energy_production(T[i], rho[i]))
                #print(L[i])

                rho[i+1] = self.rho(P[i], T[i])
                epsilon[i+1] = self.energy_production(T[i], rho[i])
                m[i+1] = m[i] - dm

        if self.end_step == 0:
            self.end_step = n

        return r, P, L, T, rho, epsilon, m, nabla, nabla_rad, nabla_ad, F_C, F_R, F

    def plot(self, r, P, L, T, rho, m):
        """
        Takes the results from the euler integration and plots everything vs. mass.
        All in units of their solar equivalent value, so the suns radius = 1, suns mass = 1, etc.
        (Except for temperature)
        """
        fig = plt.figure(figsize=(14, 8))

        plt.title("Radius")
        plt.xlabel("$\\frac{m}{m_{\\mathrm{sun}}}$")
        plt.ylabel("$\\frac{r}{R_{\\mathrm{sun}}}$")
        plt.plot(m[0 : self.end_step]/1.989e30, r[0 : self.end_step]/6.96e8)
        #plt.plot(m/1.989e30, r/6.96e8)

        plt.tight_layout()
        plt.show()

        plt.title("Luminosity")
        plt.xlabel("$\\frac{r}{R_{\\mathrm{sun}}}$")
        plt.ylabel("$\\frac{L}{L_{\\mathrm{sun}}}$")
        plt.plot(r[0 : self.end_step] / 6.96e8, L[0 : self.end_step] / 3.846e26)
        #plt.plot(m / 1.989e30, L / 3.846e26)

        plt.tight_layout()
        plt.show()

        plt.title("Temperature")
        plt.xlabel("$\\frac{m}{M_{\\mathrm{sun}}}$")
        plt.ylabel("T[MK]")
        plt.plot(m[0 : self.end_step] / 1.989e30, T[0 : self.end_step]*1e-6)
        #plt.plot(m / 1.989e30, T*1e-6)

        plt.tight_layout()
        plt.show()

        plt.title("Density")
        plt.xlabel("$\\frac{m}{M_{\\mathrm{sun}}}$")
        plt.ylabel("$\\frac{\\rho}{\\rho_{\\mathrm{sun}}}$")
        plt.plot(m[0 : self.end_step] / 1.989e30, rho[0 : self.end_step]/ 1.408e3)
        #plt.plot(m / 1.989e30, rho/ 1.408e3)
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

    def plot_cross_section(self, R_values, L_values, F_C_list):
        # -------------------------------------------------------------------------------------------------------
        # Assumptions:
        # -------------------------------------------------------------------------------------------------------
        # * R_values is an array of radii [unit: R_sun]
        # * L_values is an array of luminosities (at each r) [unit: L_sun]
        # * F_C_list is an array of convective flux ratios (at each r) [unit: relative value between 0 and 1]
        # * n is the number of elements in both these arrays
        # * R0 is the initial radius
        # * show_every is a variable that tells you to show every ...th step (show_every = 50 worked well for me)
        # * core_limit = 0.995 (when L drops below this value, we define that we are in the core)
        # -------------------------------------------------------------------------------------------------------

        # Cross-section of star
        # (See https://stackoverflow.com/questions/9215658/plot-a-circle-with-pyplot and
        # https://stackoverflow.com/questions/27629656/label-for-circles-in-matplotlib for details)
        plt.figure()
        fig = plt.gcf()  # get current figure
        ax = plt.gca()  # get current axis

        R0 = R_values[0]
        n = len(R_values)
        show_every = 2
        core_limit = 0.995

        rmax = 1.2 * R0
        ax.set_xlim(-rmax, rmax)
        ax.set_ylim(-rmax, rmax)
        ax.set_aspect('equal')  # make the plot circular
        j = show_every
        for k in range(0, n - 1):
            j += 1
            if j >= show_every:  # don't show every step - it slows things down
                if (L_values[k] > core_limit):  # outside core
                    if (F_C_list[k] > 0.0):  # convection
                        circR = plt.Circle((0, 0), R_values[k], color='red', fill=False)
                        ax.add_artist(circR)
                    else:  # radiation
                        circY = plt.Circle((0, 0), R_values[k], color='yellow', fill=False)
                        ax.add_artist(circY)
                else:  # inside core
                    if (F_C_list[k] > 0.0):  # convection
                        circB = plt.Circle((0, 0), R_values[k], color='blue', fill=False)
                        ax.add_artist(circB)
                    else:  # radiation
                        circC = plt.Circle((0, 0), R_values[k], color='cyan', fill=False)
                        ax.add_artist(circC)
                j = 0
        circR = plt.Circle((2 * rmax, 2 * rmax), 0.1 * rmax, color='red',
                       fill=True)  # These are for the legend (drawn outside the main plot)
        circY = plt.Circle((2 * rmax, 2 * rmax), 0.1 * rmax, color='yellow', fill=True)
        circC = plt.Circle((2 * rmax, 2 * rmax), 0.1 * rmax, color='cyan', fill=True)
        circB = plt.Circle((2 * rmax, 2 * rmax), 0.1 * rmax, color='blue', fill=True)
        ax.legend([circR, circY, circC, circB],
                  ['Convection outside core', 'Radiation outside core', 'Radiation inside core',
                   'Convection inside core'])  # only add one (the last) circle of each colour to legend
        plt.xlabel('')
        plt.ylabel('')
        plt.title('Cross-section of star')
        plt.tight_layout()

        # Show all plots
        plt.show()

if __name__ == "__main__":
    f = SolarCore()

    #f.energy_production(1e8, 1.62e5)

    r, P, L, T, rho, epsilon, m, nabla, nabla_rad, nabla_ad, F_C, F_R, F = f.euler()


    f.plot(r, P, L, T, rho, m)
    #f.plot_cross_section(r / f.R_sun, L/f.L_sun, F_C/F)

    plt.plot(r[0 : f.end_step] / f.R_sun, F_C[0 : f.end_step]/F[0 : f.end_step], r[0 : f.end_step] / f.R_sun, F_R[0 : f.end_step]/F[0 : f.end_step])
    plt.show()

    plt.title("Temperature gradients")
    plt.xlabel("$\\frac{r}{R_{\\mathrm{sun}}}$")
    #plt.ylabel("")
    plt.plot(r[0 : f.end_step] / f.R_sun, nabla[0 : f.end_step], r[0 : f.end_step] / f.R_sun, nabla_rad[0 : f.end_step], r[0 : f.end_step] / f.R_sun, nabla_ad[0 : f.end_step])
    plt.yscale("symlog")
    plt.ylim(0, 1e3)
    plt.legend(["nabla", "nabla_rad", "nabla_ad"])
    #plt.tight_layout()
    plt.show()
