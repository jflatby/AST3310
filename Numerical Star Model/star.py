import numpy as np
import scipy.constants as const
from astropy import constants as astroconst
from astropy import units as units
import matplotlib.pyplot as plt
import matplotlib as mplt
import sys
from scipy.interpolate import interp2d
import time
plt.style.use("ggplot")

class Star:

    def __init__(self):
        ## Constants
        self.sigma = const.Stefan_Boltzmann
        self.k = const.Boltzmann
        self.m_u = const.atomic_mass
        self.G = const.gravitational_constant
        self.NA = const.Avogadro
        self.L_sun = astroconst.L_sun.value
        self.R_sun = astroconst.R_sun.value
        self.M_sun = astroconst.M_sun.value
        self.a = 4 * self.sigma / const.speed_of_light

        ## Mass Fractions
        X = 0.7
        Y3 = 1e-10
        Y = 0.29
        Z = 0.01
        Z_Li = 1e-13
        Z_Be = 1e-13

        ## Initial Values
        self.L0 = 1.0 * self.L_sun  # [W]
        self.R0 = 1.0 * self.R_sun  # [m]
        self.M0 = 1.0 * self.M_sun  # [kg]
        self.T0 = 1.0 * 5770  # [K]
        self.rho0 = 1.0 * 1.42e-7 * 1.408e3  # [kg/m^3]

        """
        Best Model:
        self.L0 = 1.0 * self.L_sun  # [W]
        self.R0 = 0.9 * self.R_sun  # [m]
        self.M0 = 0.95 * self.M_sun  # [kg]
        self.T0 = 0.7 * 5770  # [K]
        self.rho0 = 54 * 1.42e-7 * 1.408e3  # [kg/m^3]
        """

        self.PP_values = []
        self.recorded_pp = False

        ## molecular average weight
        self.mu = 1 / (2 * X + 3 / 4 * Y + 9 / 14 * Z)

        self.c_p = 5 / 2 * self.k / (self.mu * self.m_u)

        self.end_step = 0
        self.unstable = True
        self.variable_step = True

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

        return 10**kappa[0] * 0.1

    def rho(self, P, T): #From P = P_G + P_rad
        rho = self.mu * self.m_u / (self.k * T) * (P - (self.a / 3) * T**4) # P_G + P_R
        return rho

    def P(self, rho, T):
        P = rho*self.k*T/(self.mu*self.m_u) + (self.a / 3) * T**4
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
        Q_17Be = 0.14*MeVtoJ #(0.14 + (1.02 + 6.88) + 3.00)*MeVtoJ #0.14*MeVtoJ

        eps = Q_pp * r_pp + Q_33 * r_33 \
                  + Q_34 * r_34 + Q_e7 * r_e7 + Q_17Li * r_17Li \
                  + Q_17Be * r_17Be
        #print(f"{Q_pp * r_pp*rho} + {Q_33 * r_33*rho} + {Q_34 * r_34*rho} + {Q_e7 * r_e7*rho} + {Q_17Li * r_17Li*rho} + {Q_17Be * r_17Be*rho}")

        PPI = (Q_pp * r_pp + Q_33 * r_33 )/eps
        PPII = (Q_34 * r_34 + Q_e7 * r_e7 + Q_17Li * r_17Li)/eps
        PPIII = (Q_17Be * r_17Be)/eps

        #print(PPI, PPII, PPIII)

        if not self.recorded_pp:
            self.PP_values.append([PPI, PPII, PPIII])
            self.recorded_pp = True

        return eps

    def dr(self, r, rho):
        return 1/(4 * np.pi * r**2 * rho)

    def dP(self, r, m):
        return - self.G * m / (4 * np.pi * r ** 4)

    def dT(self, r, T, L, rho):
        return -3 * self.opacity(T, rho) * L / (256 * np.pi ** 2 * self.sigma * r ** 4 * T ** 3)

    def nabla_stable(self, T, P, L, m, kappa):
        return 3*kappa * L * P / (64 * np.pi * self.sigma * self.G * m * T**4)

    def nabla_ad(self, P, T, rho):
        delta = 1
        return P * delta / (T * rho * self.c_p)

    def nabla_star(self, T, rho, kappa, H_P, g, nabla_stable, nabla_ad):
        U = (64 * self.sigma * T ** 3) / (3 * kappa * rho ** 2 * self.c_p) * np.sqrt(H_P / (g))

        alpha = 1
        # Mixing length
        l_m = alpha * H_P

        K = 4 / l_m ** 2

        # solve to find xi
        roots = np.roots(
            np.array([1, U / l_m ** 2, U ** 2 / l_m ** 2 * K, -U / l_m ** 2 * (nabla_stable - nabla_ad)]))
        for root in roots:
            if np.imag(root) == np.min(np.abs(np.imag(roots))):
                xi = np.real(root)
                break
        # use xi to calculate nabla*
        nabla = xi ** 2 + U * K * xi + nabla_ad

        return nabla

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
        start_time = time.time()
        n = int(1e5)
        r, P, L, T, rho, epsilon, m = np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)

        r[0], P[0], L[0], T[0], rho[0], epsilon[0], m[0] = self.R0, self.P0, self.L0, self.T0, self.rho0, self.energy_production(self.T0, self.rho0), self.M0
        nabla, nabla_stable, nabla_ad, F_C, F_R, F = np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)

        nabla_stable[0], nabla_ad[0] = self.nabla_stable(T[0], P[0], L[0], m[0], self.opacity(T[0], rho[0])), self.nabla_ad(P[0], T[0], rho[0])

        for i in range(n-1):
            sys.stdout.write('Progress: %4.4f %s\r' % (100 - (m[i] / self.M0) * 100, '%'))
            sys.stdout.flush()


            if r[i] < 1e-4 or L[i] < 1e-4 or rho[i] < 1e-4 or P[i] < 1e-4 or T[i] < 1e-4:
                print("----- Other value below zero -----")
                print("Step:", i)
                print("Mass:", m[i])
                print(f"{m[i]*100/self.M0:.2f}% M0")
                print("Radius:", r[i])
                print(f"{r[i]*100/self.R0:.2f}% R0")
                print("Luminosity:", L[i])
                print(f"{L[i]*100/self.L0:.2f}% L0")
                print("Density:", rho[i])
                print("Pressure:", P[i])
                print("Temperature:", T[i])
                self.end_step = i
                break
            elif m[i] < 1e-4:
                print("----- Reached zero mass -----")
                print("Final values:")
                print("Step:", i)
                print("Mass:", m[i])
                print("Radius:", r[i])
                print(f"{r[i]*100/self.R0:.2f}% R0")
                print("Luminosity:", L[i])
                print(f"{L[i]*100/self.L0:.2f}% L0")
                print("Density:", rho[i])
                print("Pressure:", P[i])
                print("Temperature:", T[i])
                self.end_step = i
                break
            else:
                kappa = self.opacity(T[i], rho[i])
                g = self.G * m[i] / r[i] ** 2
                H_P = P[i] / (g * rho[i])

                nabla_stable[i] = self.nabla_stable(T[i], P[i], L[i], m[i], kappa)
                nabla_ad[i] = self.nabla_ad(P[i], T[i], rho[i])

                self.unstable = nabla_stable[i] > nabla_ad[i]

                F[i] = self.total_flux(L[i], r[i])

                if self.unstable:
                    nabla[i] = self.nabla_star(T[i], rho[i], kappa, H_P, g, nabla_stable[i], nabla_ad[i])
                    F_R[i] = self.F_R(T[i], kappa, rho[i], H_P, nabla[i])
                    F_C[i] = self.F_C(F[i], F_R[i])
                    dT = -self.G * m[i] * T[i] * nabla[i] / (4 * np.pi * r[i]**4 * P[i])
                else:
                    nabla[i] = nabla_stable[i]
                    F_R[i] = F[i]
                    dT = self.dT(r[i], T[i], L[i], rho[i])

                if self.variable_step:
                    f = np.abs(np.array([self.dr(r[i], rho[i]), self.dP(r[i], m[i]), self.energy_production(T[i], rho[i]), dT]))
                    V = np.array([r[i], P[i], L[i], T[i]])
                    p = 0.01
                    dm = np.min(p * V / f)
                    if dm < 1e19:
                        dm = 1e19


                r[i+1] = r[i] - self.dr(r[i], rho[i]) * dm
                P[i+1] = P[i] - self.dP(r[i], m[i]) * dm
                L[i+1] = L[i] - self.energy_production(T[i], rho[i]) * dm
                T[i+1] = T[i] - dT * dm
                m[i+1] = m[i] - dm

                self.recorded_pp = False

                rho[i+1] = self.rho(P[i+1], T[i+1])
                epsilon[i+1] = self.energy_production(T[i+1], rho[i+1])


        if self.end_step == 0:
            self.end_step = n

        print(f"Done. Time elapsed: {time.time() - start_time:.2f}s")
        return r, P, L, T, rho, epsilon, m, nabla, nabla_stable, nabla_ad, F_C, F_R, F

    def plot(self, r, P, L, T, rho, m):
        """
        Takes the results from the euler integration and plots the results.
        All in units of their solar equivalent value, so the suns radius = 1, suns mass = 1, etc.
        (Except for temperature)
        """
        #fig = plt.figure()

        #ax1 = plt.subplot(311)
        plt.title("Luminosity")
        plt.xlabel("$\\frac{r}{R_0}$")
        plt.ylabel("$\\frac{L}{L_0}$")
        plt.plot(r[0 : self.end_step] / self.R0, L[0 : self.end_step] / self.L0)
        plt.tight_layout()
        plt.show()

        #ax2 = plt.subplot(312, sharex=ax1)
        plt.title("Mass")
        plt.xlabel("$\\frac{r}{R_0}$")
        plt.ylabel("$\\frac{m}{m_0}$")
        plt.plot(r[0: self.end_step] / self.R0, m[0: self.end_step] / self.M0)
        plt.tight_layout()
        plt.show()

        #ax3 = plt.subplot(313, sharex=ax1)
        plt.title("Temperature")
        plt.xlabel("$\\frac{r}{R_0}$")
        plt.ylabel("T[MK]")
        plt.plot(r[0: self.end_step] / self.R0, T[0: self.end_step] * 1e-6)
        plt.tight_layout()

        #plt.setp(ax1.get_xticklabels(), visible=False)
        #plt.setp(ax2.get_xticklabels(), visible=False)
        #plt.setp(ax3.get_xticklabels(), visible=False)

        plt.show()

        #ax4 = plt.subplot(2, 1, 1)
        plt.title("Density")
        plt.xlabel("$\\frac{r}{R_0}$")
        plt.ylabel("$\\frac{\\rho}{\\rho_0}$")
        plt.plot(r[0: self.end_step] / self.R0, rho[0: self.end_step] / self.rho0)
        plt.yscale("symlog")
        plt.tight_layout()
        plt.show()

        #ax5 = plt.subplot(2, 1, 2, sharex=ax4)
        plt.title("Pressure")
        plt.xlabel("$\\frac{r}{R_0}$")
        plt.ylabel("$\\frac{P}{P_0}$")
        plt.plot(r[0: self.end_step] / self.R0, P[0: self.end_step] / self.P0)
        plt.yscale("symlog")
        plt.tight_layout()
        #plt.setp(ax4.get_xticklabels(), visible=False)
        plt.show()

        ## Plot PP energy levels
        """
        PP = np.transpose(mystar.PP_values)
        print(PP)
        plt.title("PP energy levels")
        #plt.plot(r[0 : mystar.end_step+1] / mystar.R0, epsilon[0 : mystar.end_step+1]/np.max(epsilon))
        plt.plot(r[0 : mystar.end_step+1]/ mystar.R0, PP[0], r[0 : mystar.end_step+1] / mystar.R0, PP[1], r[0 : mystar.end_step+1] / mystar.R0, PP[2])
        plt.legend(["PPI", "PPII", "PPIII"])
        plt.show()

        plt.plot(r[0 : mystar.end_step+1] / mystar.R0, PP[0])#, r[0 : mystar.end_step+1] / mystar.R0, PP[1]/epsilon[0:mystar.end_step+1])
        plt.legend(["PPI"])
        plt.show()
        plt.plot(r[0 : mystar.end_step+1] / mystar.R0, PP[1])
        plt.legend(["PPII"])
        plt.show()
        plt.plot(r[0 : mystar.end_step+1] / mystar.R0, PP[2])
        plt.legend(["PPIII"])
        plt.show()
        """

        ## Plot Convective and Radiative Flux
        """
        plt.title("Radiative and convective flux")
        plt.xlabel("$\\frac{r}{R_{\\mathrm{sun}}}$")
        plt.ylabel("F")
        plt.plot(r[0 : mystar.end_step] / mystar.R_sun, F_C[0 : mystar.end_step]/F[0 : mystar.end_step], r[0 : mystar.end_step] / mystar.R_sun, F_R[0 : mystar.end_step]/F[0 : mystar.end_step])
        plt.legend(["Convective Flux", "Radiative Flux"])
        plt.show()

        plt.title("Temperature gradients")
        plt.xlabel("$\\frac{r}{R_{\\mathrm{sun}}}$")
        #plt.ylabel("")
        plt.plot(r[0 : mystar.end_step] / mystar.R_sun, nabla[0 : mystar.end_step], r[0 : mystar.end_step] / mystar.R_sun, nabla_stable[0 : mystar.end_step], r[0 : mystar.end_step] / mystar.R_sun, nabla_ad[0 : mystar.end_step])
        plt.yscale("symlog")
        plt.ylim(0, 1e3)
        plt.legend(["nabla", "nabla_stable", "nabla_ad"])
        #plt.tight_layout()
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
        ax = plt.gca()  # get current axis

        n = len(R_values)
        last_conv_step = 0
        first_core_step = 0
        show_every = 10
        core_limit = 0.995

        rmax = 1.2 * R_values[0]
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
                        last_conv_step = k
                    else:  # radiation
                        circY = plt.Circle((0, 0), R_values[k], color='yellow', fill=False)
                        ax.add_artist(circY)
                else:  # inside core
                    if(first_core_step == 0):
                        first_core_step = k
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
        plt.xlabel(f'Outer convection layer: {(R_values[0] - R_values[last_conv_step])/R_values[0] * 100:.1f} % of $R_0$. Core: {(R_values[first_core_step])/R_values[0] * 100:.1f} % of $R_0$')
        plt.ylabel('')
        plt.title('Cross-section of star')
        plt.tight_layout()

        # Show all plots
        plt.show()

if __name__ == "__main__":
    mystar = Star()

    r, P, L, T, rho, epsilon, m, nabla, nabla_stable, nabla_ad, F_C, F_R, F = mystar.euler()

    #mystar.plot(r, P, L, T, rho, m)
    mystar.plot_cross_section(r[:mystar.end_step] / mystar.R_sun, L[:mystar.end_step]/mystar.L_sun, [F_C[i]/F[i] if F[i] else 0 for i in range(len(F_C[:mystar.end_step]))])