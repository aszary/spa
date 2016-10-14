#! /usr/bin/env python
__author__="andrzej"
__date__ ="$2015-02-03 18:11:55$"
################################################################################
from math import pi

################################################################################
class CGS:

    pi = pi
    SOLAR_MASS = 1.98855e33                 # solar mass
    G = 6.6725985e-8                        # gravitational constant
    h_bar = 1.0545726663e-27                # Planck's constant divided by 2 pi
    h = 6.626075540e-27                     # Planck's constant
    c = 29979245800.0                       # speed of light
    e = 4.803206814e-10                     # elementary charge in  statcoulombs (esu)
    q_i = 26. * 4.803206814e-10             # iron ion charge (Fe^26)
    m = 9.109389754e-28                     # electron mass
    m_e = 9.109389754e-28                   # electron mass
    m_p = 1.672623110e-24                   # proton mass
    m_n = 1.674928610e-24                   # neutoron mass
    m_i = 26. * m_p + 30. * m_n             # iron ion mass (Fe^26)
    B_crit = m**2 * c**3 / (e * h_bar)      # critical magnetic field
    b = 0.28977685                          # Wien's displacment constant
    k = 1.38065812e-16                      # Boltzmann constant
    sigma = 5.6705119e-5                    # Stefan-Boltzman constant [pi**2. * k**4. / (60. * h_bar**3. * c**2.)]
    sigma_th = 6.6524586e-25                # Thompson cross section [8. * pi / 3. * (e**2. / (m * c**2.))**2.]
    alpha = 7.297352537650e-3               # Fine structure constant [e**2. / (h_bar * c)]
    a_0 = h_bar / (m * c * alpha)           # Bhor radius [52.9177e-10]
    lambda_c = 2.4263102175e-10              # Electron Compton wavelength [h / (m * c)]

class SI:

    pi = pi
    SOLAR_MASS = 1.98855e30                 # solar mass
    G = 6.67428e-11                         # gravitational constant
    h_bar = 1.054571628e-34                 # Planck's constant divided by 2 pi
    h = 6.62606896e-34                      # Planck's constant
    c = 299792458.0                         # speed of light
    e = 1.602176487e-19                     # elementary charge in  statcoulombs (esu)
    m = 9.10938215e-31                      # electron mass
    epsilon_0 = 8.854187817e-12
    mu_0 = 12.566370614e-7
    b = 2.8977685e-3                        # Wien's displacment constant
    k =	1.3806504e-23                       # Boltzmann constant
    alpha = 7.2973525376e-3                 # Fine structure constant [e**2. / (h_bar * c)]
    lambda_c = 2.4263102175e-12             # Electron Compton wavelength [h / (m * c)]


################################################################################
def main():

    c = CGS
    print 79*"-"

    print "CGS units:"
    for var in dir(CGS):
        if not var.startswith("_"):
            print "%s=%e"%(var,getattr(CGS,var))

    print 79*"-"
    print 79*"-"

    print "SI units:"
    for var in dir(SI):
        if not var.startswith("_"):
            print "%s=%e"%(var,getattr(SI,var))

    print 79*"-"

    print 'Bye'
    pass

################################################################################
if __name__ == '__main__':
    main()
