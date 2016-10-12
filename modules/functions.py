#! /usr/bin/env python
__author__ = 'aszary'
from math import cos, exp, exp

from const.const import CGS as CO


class Functions:

    def __init__(self, r=10e5, m=1.4):
        """
        :param r: neutron star radius in centimeters
        :param m: neutron star mass in solar masses
        """
        self.CO = CO
        self.r = r
        self.m = m * CO.SOLAR_MASS
        self.gr = (1 - 2. * CO.G * self.m / (self.r * CO.c ** 2.)) ** 0.5

    def t_s(self, t_inf):
        """
        :param t_inf: temperature obtained from the BB fit
        :return: redshifted temperature.
        """
        return t_inf * self.gr ** (-1.)

    def r_pc(self, r_perp_inf, f):
        """
        :param r_perp_inf: radius estimated assuming isotropic radiation
        :param f: geometrical factor (depends on alpha, number of spots, etc.)
        :return: radius of the polar cap/s
        """
        return self.gr * f ** (-0.5) * r_perp_inf

    @staticmethod
    def kev_to_mk(t_kev):
        """
        :param t_kev: temperature in kiloelectronoVolts
        :return: temperature in 10^6 kelvin
        """
        return t_kev * 1e3 / 8.61734315e-5 / 1e6

    @staticmethod
    def ev_to_k(t_ev):
        """
        :param t_ev: temperature in electronoVolts
        :return: temperature in kelvin
        """
        return t_ev / 8.61734315e-5


    @staticmethod
    def radius_bbodyrad(norm, dist_kpc):
        """
        :param norm: normalization in bbodyrad fit (Xspec)
        :param dist_kpc: distance to the source in kpc
        :return: radius in meters (r_perp_inf)
        """
        r_km = (norm * (dist_kpc / 10.) ** 2.) ** 0.5
        return r_km * 1e3

    @staticmethod
    def luminosity(t6, rm):
        """
        The polar cap luminosity. Note factor of 4 difference from a sphere.
        :param t6: temperature in 10^6 kelvin
        :param rm: polar cap radius in meters
        :return: blackbody luminosity
        """
        return CO.sigma * (t6 * 1e6) ** 4. * CO.pi * (rm * 1e2) ** 2.

    @staticmethod
    def luminosity_cgs(t, r):
        """
        The polar cap luminosity. Note factor of 4 difference from a sphere.
        :param t: temperature in kelvin
        :param r: polar cap radius in centimeters
        :return: blackbody luminosity
        """
        return CO.sigma * t ** 4. * CO.pi * r ** 2.

    @staticmethod
    def bs_crit(t):
        """
        Magnetic field strength fulfilling the critical condition Medin & Lai (2007) - see critical_medin
        :param t: temperature in kelvin
        :return: magnetic field strength in G
        """
        return (t / 1e6 / 2.) ** (4./3.) * 1e14

    @staticmethod
    def ts_crit(b):
        """
        Surface temperature  fulfilling the critical condition Medin & Lai (2007) - see critical_medin
        :param b: magnetic field strength in G
        :return: surface temperature in kelvin
        """
        return 2.0 * 1e6 * (b / 1e14) ** 0.75  # 0.74 in a fit

    @staticmethod
    def e_dot(p, pdot):
        """
        spin-down luminosity
        :param p: pulsar period
        :param pdot: pulsar period derivative
        :return: spin-down luminosity
        """
        return 3.95e31 * (pdot/1e-15) / p ** 3.

    @staticmethod
    def v_vg(b_s, p, h_perp, alpha=0.):
        """
        vacuum gap potential
        :param b_s: surface magnetic field
        :param p: pulsar period
        :param h_perp: spark half-width
        :param alpha: inclination angle between rotation and magnetic axes (default: align rotator)
        :return: maximum potential assuming vacuum in a spark
        """
        return 4. * CO.pi * b_s * cos(alpha) / (CO.c * p) * h_perp ** 2.

    @staticmethod
    def delta_v(eta, b_s, p, h_perp, alpha=0.):
        """
        gap potential
        :param eta, screening factor
        :param b_s: surface magnetic field
        :param p: pulsar period
        :param h_perp: spark half-width
        :param alpha: inclination angle between rotation and magnetic axes (default: align rotator)
        :return: maximum potential assuming vacuum in a spark
        """
        return eta * 4. * CO.pi * b_s * cos(alpha) / (CO.c * p) * h_perp ** 2.


    def r_dp(self, p):
        """
        Polar cap radius assuming purely dipolar configuration of the magnetic field at the surface
        :param p: pulsar period
        :return: radius of thr polar cap in centimeters
        """
        return (2. * CO.pi * self.r ** 3. / (CO.c * p)) ** 0.5

    def b_d(self, p, pdot):
        """
        Dipolar component of the magnetic field at the polar cap
        :param p: period
        :param pdot: period derivative
        :return: magnetic field at the polar cap
        """
        return 2.02e12 * (p * pdot / 1e-15) ** 0.5


    @staticmethod
    def lph_erber(gamma, b_perp):
        """ Photon mean free path Erber 1966
        :param gamma: photon energy / mc^2
        :param b_perp: magnetic field strength (perpendicular component to the motion direction)
        :return: photon mean free path
        """
        try:
            chi = gamma / 2. * b_perp / CO.B_crit
            l_ph = 4.4 / (CO.e ** 2. / (CO.h_bar * CO.c)) * CO.h_bar / (CO.m * CO.c) * CO.B_crit / b_perp * exp(4. / (3. * chi))
        except:
            l_ph = 1e300
        return l_ph

    @staticmethod
    def lph_quantum(gamma, re):
        """ Photon mean free path in strong magnetic fields beta_q=B/B_crit > 0.2
        :param gamma: photon energy / mc^2
        :param re: curvature radius
        :return: photon mean free path
        """
        l_ph = re * (2. / gamma)
        return l_ph


def main():
    f = Functions()
    print f.t_s(2.)
    print 'Bye'


if __name__ == '__main__':
    main()