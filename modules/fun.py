__author__ = 'aszary'
#import sys
from math import factorial
from copy import deepcopy

import numpy as np
import sympy as sy
from scipy.optimize import leastsq
from interpolate import least_sq_err, least_sq

from functions import Functions
F = Functions()
import peakutils as pk


def rlc(p):
    """
    :param p: pulsar period
    :return: radius of the light cylinder [in centimeters]
    """
    return F.CO.c * p / (2. * np.pi)

def theta_max(z, p):
    """ Note this is not an opening angle (Note 1.5 differecence E.g 3.29 in a handbook)
    :param z: distance from the star's center [in stellar radius]
    :param p: pulsar period
    :return: last open magnetic field line for a given distance from the star's center and pulsar period [in radians]
    """
    return np.arcsin(np.sqrt(z * F.r / rlc(p)))

def rho(z, p):
    """
    works only for small alpha!
    :param z:  distance from the star's center [in stellar radius]
    :param p: pulsar period
    :return: opening angle for a given distance from the star's center and pulsar period [in radians]
    """
    return 1.5 * theta_max(z, p)

def rho_sy(theta):
    """
    Calculates an opening angle for a given magnetic field line
    :param theta: coordinate of last opened magnetic field line
    :return: opening angle
    """
    rho = sy.symbols('rho')

    expr = -3. / (2. * sy.tan(rho)) + sy.sqrt(2.+(3. / (2. * sy.tan(rho))) ** 2)
    expr2 = -3. / (2. * sy.tan(rho)) - sy.sqrt(2.+(3. / (2. * sy.tan(rho))) ** 2)

    res = sy.solve(sy.Eq(expr, theta), rho)
    res2 = sy.solve(sy.Eq(expr2, theta), rho)

    # not very smart (take a break pls, no!)
    try:
        if res[0] > 0.:
            return res[0]
        else:
            return res2[0]
    except:
        return res2[0]


def dipolar_cap_radius(p):
    """
    :param p: pulsar period
    :return: the polar cap radius assuming dipolar configuration of the magnetic field [in centimeters]
    """
    return np.sqrt(F.r ** 3. / rlc(p))

def b_eq(p, dot_p):
    """
    :param p: pulsar period
    :param dot_p: period derivative
    :return: surface magnetic flux density at the equator [in Gauss]
    """
    return 3.2 * 1e19 * (p * dot_p) ** 0.5

def b_d(p, dot_p):
    """
    :param p: pulsar period
    :param dot_p: period derivative
    :return: dipolar component of the magnetic field at the polar cap [in Gauss]
    """
    return 2.02 * 1e12 * p ** 0.5 * (dot_p / 1e-15) ** 0.5

def to_radians(angle):
    """
    :param angle: angle in degrees
    :return: angle in radians
    """
    return angle / 180. * np.pi

def to_degrees(angle):
    """
    :param angle: angle in radians
    :return: angle in degrees
    """
    return angle / np.pi * 180.

def spherical_cartesian(spherical):
    """
    :param spherical: spherical coordinates
    :return: cartesian coordinates
    """
    x = spherical[0] * np.sin(spherical[1]) * np.cos(spherical[2])
    y = spherical[0] * np.sin(spherical[1]) * np.sin(spherical[2])
    z = spherical[0] * np.cos(spherical[1])
    return x, y, z

def cartesian_spherical(cartesian):
    """
    :param cartesian: cartesian coordinates
    :return: spherical coordinates
    """
    r = np.sqrt(cartesian[0] ** 2. + cartesian[1] ** 2. + cartesian[2] ** 2.)
    theta = mod(np.arctan2(np.sqrt(cartesian[0] ** 2. + cartesian[1] ** 2.), cartesian[2]))
    phi = mod(np.arctan2(cartesian[1], cartesian[0]))
    return np.array([r, theta, phi])

def vec_spherical_cartesian(pos_sph, vec_sph):
    """
    :param pos_sph: position of the vector in spherical coordinates
    :param vec_sph: spherical coordinates of the vector
    :return: cartesian components of the vector
    """
    r = pos_sph[0]
    theta = pos_sph[1]
    phi = pos_sph[2]

    b_r = vec_sph[0]
    b_theta = vec_sph[1]
    b_phi = vec_sph[2]

    # DONE b_z component sin(phi) --> sin(theta)
    b_x = b_r * np.sin(theta) * np.cos(phi) + b_theta * np.cos(theta) * np.cos(phi) - b_phi * np.sin(phi)
    b_y = b_r * np.sin(theta) * np.sin(phi) + b_theta * np.cos(theta) * np.sin(phi) + b_phi * np.cos(phi)
    b_z = b_r * np.cos(theta) - b_theta * np.sin(theta)
    return b_x, b_y, b_z

def mod(x, min=0., max=2.*np.pi):
    """
    :param x: an angle
    :param min: minimum value for the angle
    :param max: maximum value for the angle
    :return: min <= angle <= max
    """
    while x < min:
        x = x + max
    while x > max:
        x = x - max
    return x

def find_center(x_, y_, z_):
    """
    :param x_
    :param y_
    :param z_
    :return: cartesian coordinate of the polar cap center (centroid)
    """
    return np.mean(x_), np.mean(y_), np.mean(z_)

def distance(a, b):
    """
    Calculates distance between two points a and b
    :param a: first point
    :param b: second point
    :return: distance between a and b
    """
    return np.sqrt((b[0] - a[0]) ** 2. + (b[1] - a[1]) ** 2. + (b[2] - a[2]) ** 2.)

def distance2D(a, b):
    """
    Calculates distance between two points a and b in 2D
    :param a: first point
    :param b: second point
    :return: distance between a and b
    """
    return np.sqrt((b[0] - a[0]) ** 2. + (b[1] - a[1]) ** 2.)


def vectors_angle(a, b):
    """
    Calculates angle between two vectors in radians
    :param a: first vector
    :param b: second vector
    :return: angle between two vectors [in rad]
    """
    ab_vec = a[0]*b[0] + a[1]*b[1] + a[2] * b[2]
    a_ = np.sqrt(a[0] ** 2. + a[1] ** 2. + a[2] ** 2.)
    b_ = np.sqrt(b[0] ** 2. + b[1] ** 2. + b[2] ** 2.)
    ab = a_ * b_
    # to avoid some numerical errors
    if ab < ab_vec:
        ab = ab_vec
    alpha = np.arccos(ab_vec/ab)
    return alpha

def in_km(d):
    """
    converts distance from one in stellar radius units to kilometers
    :param d: distance in stellar radius
    :return: distance in kilometers
    """
    return np.array(d) * F.r / 1e5

def in_m(d):
    """
    converts distance from one in stellar radius units to meters
    :param d: distance in stellar radius
    :return: distance in kilometers
    """
    return np.array(d) * F.r / 1e2

def gaussian_2D(x, y, x_0, y_0, sigma_x, sigma_y, a=1.):
    """
    Used, for instance, to calculate sub-pulse intensity based on spark location
    :param x: observer's location
    :param y: observer's location
    :param x_0: spark location
    :param y_0: spark location
    :param sigma_x: x variance
    :param sigma_y: y variance
    :param a: amplitude modification
    :return:
    """
    return a * np.exp(-((x - x_0)**2 / (2. * sigma_x**2) + (y - y_0)**2 / (2. * sigma_y**2)))


def get_single_pulses(pulses, ratio=4./3.):
    """

    :param pulses: single pulses flux
    :param ratio: aspect ratio of resulting matrix (for imshow)
    :return:
    """
    yl, xl = pulses.shape
    ma = max([yl, xl])
    # new sizes
    xn = ma
    yn = int(ma / ratio)
    # modifications
    xm = float(xn) / xl
    ym = float(yn) / yl + 1
    profile_ = np.zeros([yn, xn])
    for i in xrange(yn):
        for j in xrange(xn):
            profile_[i][j] = pulses[int(i/ym)][int(j/xm)]
    y_max = int(i/ym)
    return profile_, y_max

def get_maxima_old(pulses, comp_num, pthres=0.5, sthres=0.1, smooth=True):
    max_x_ = []
    max_y_ = []
    for i in xrange(comp_num):
        max_x_.append([])
        max_y_.append([])
    size = len(pulses[0])
    ns = size / comp_num

    sum_ = np.zeros([comp_num, len(pulses)])
    max_ = np.zeros([comp_num])
    pu_ = []
    for i in xrange(len(pulses)):
        # split window based on number of components
        pu_.append([])
        j = -1
        for j in xrange(comp_num-1):
            pu_[-1].append(pulses[i][j*ns : (j+1)*ns])
        pu_[-1].append(pulses[i][(j+1)*ns : ])

    # smooth data
    sm_ = deepcopy(pu_)
    for i in xrange(len(pulses)):
        for j in xrange(comp_num):
            if smooth is True:
                smoothed = savitzky_golay(pu_[i][j], 71, 3)
            else:
                smoothed = pu_[i][j]
            #base = pk.baseline(smoothed, 3)
            #smoothed -= base
            #smoothed = pu_[i][j]  # smoothed disabled
            for k in xrange(len(pu_[i][j])):
                sm_[i][j][k] = smoothed[k]
            sum_[j][i] = np.sum(pu_[i][j])

    for j in xrange(comp_num):
        max_[j] = np.max(sum_[j])

    for i in xrange(len(pulses)):
        for j in xrange(comp_num):
            # find one peak
            if sum_[j][i] > sthres * max_[j]:
                pind = pk.indexes(sm_[i][j], min_dist=size/2, thres=pthres)
                if len(pind) >= 1:
                    new_ind = pk.interpolate(np.arange(len(sm_[i][j])), sm_[i][j], ind=pind, width=20, func=pk.centroid)
                    if np.isnan(new_ind[0]):
                        max_x_[j].append(pind[0] + j * (ns+1))  # TODO +1 hmm?
                        max_y_[j].append(i)
                    else:
                        max_x_[j].append(new_ind[0] + j * (ns+1))
                        max_y_[j].append(i)
                    import matplotlib.pyplot as pl
                    pl.plot(sm_[i][j])
                    pl.show()
                else:
                    pass
                    """
                    import matplotlib.pyplot as pl
                    pl.plot(sm_[i][j])
                    pl.show()
                    """
    return max_x_, max_y_


def get_two_maxima_old(pulses, pthres=0.5, smooth=True):
    """
    Get two maxima based on peak detection and gaussian fitting
    """
    comp_num = 2
    max_x_ = []
    max_y_ = []
    for i in xrange(comp_num):
        max_x_.append([])
        max_y_.append([])
    size = len(pulses[0])
    x_ = np.array(range(size))

    for i in xrange(len(pulses)):
        # get maxima
        pind = pk.indexes(pulses[i], min_dist=size / 5, thres=pthres)  # 5 really?
        if len(pind) == 2:
            v0 = [pulses[i][pind[0]], float(pind[0]), size / 10., pulses[i][pind[1]], float(pind[1]), size / 10.]
            ## Error function
            errfunc = lambda v, x, y: (f2(v, x) - y)
            res = leastsq(errfunc, v0, args=(x_, pulses[i]), maxfev=1000, full_output=False)
            v = res[0]
            """
            print v0
            print v

            ga20 = f2(v0, x_)
            ga2 = f2(v, x_)
            import matplotlib.pyplot as pl
            pl.plot(pulses[i])
            pl.plot(ga2, c="red")
            pl.plot(ga20, c="green")
            pl.axvline(x=v[1])
            pl.axvline(x=v[4])
            pl.show()
            """
            max_x_[0].append(v[1])
            max_y_[0].append(float(i))
            max_x_[1].append(v[4])
            max_y_[1].append(float(i))
        elif len(pind) == 1:
            print "Warning: two maxima not found... (%d)" % len(pind)
            v0 = [pulses[i][pind[0]], float(pind[0]), size / 10.]
            ## Error function
            errfunc = lambda v, x, y: (f(v, x) - y)
            res = leastsq(errfunc, v0, args=(x_, pulses[i]), maxfev=1000, full_output=False)
            v = res[0]
            """
            ga0 = f(v0, x_)
            ga = f(v, x_)
            import matplotlib.pyplot as pl
            pl.plot(pulses[i])
            pl.plot(ga, c="red")
            pl.plot(ga0, c="green")
            pl.axvline(x=v[1])
            pl.show()
            """
        else:
            print "Warning: two maxima not found... (%d)" % len(pind)
    return max_x_, max_y_

def get_two_maxima_old2(pulses, pthres=0.5, smooth=True):
    """
    Get two maxima based on two gaussian fitting
    """
    comp_num = 2
    max_x_ = []
    max_y_ = []
    for i in xrange(comp_num):
        max_x_.append([])
        max_y_.append([])
    size = len(pulses[0])
    x_ = np.array(range(size))

    for i in xrange(len(pulses)):
        # get maxima
        v0 = [pulses[i][size/4], size/4., size / 20., pulses[i][size*3/4], size*3./4., size / 20.]
        ## Error function
        errfunc = lambda v, x, y: (f2(v, x) - y)
        res = leastsq(errfunc, v0, args=(x_, pulses[i]), maxfev=10000, full_output=False)
        v = res[0]
        #"""
        print v0
        print v
        if i == 17:
            ga20 = f2(v0, x_)
            ga2 = f2(v, x_)
            import matplotlib.pyplot as pl
            pl.plot(pulses[i])
            pl.plot(ga2, c="red")
            pl.plot(ga20, c="green")
            pl.axvline(x=v[1])
            pl.axvline(x=v[4])
            #pl.show()
            pl.savefig("output/te.pdf")
        #"""
        max_x_[0].append(v[1])
        max_y_[0].append(float(i))
        max_x_[1].append(v[4])
        max_y_[1].append(float(i))
    return max_x_, max_y_

def get_maxima2(pulses, comp_num, pthres=0.1, smooth=True):
    """
    Get maxima based on two gaussians fitting (with subcomponents)
    """
    max_x_ = []
    max_y_ = []
    for i in xrange(comp_num):
        max_x_.append([])
        max_y_.append([])

    size = len(pulses[0])
    ns = size / comp_num
    pu_ = []
    indexes = []

    for i in xrange(comp_num):
        # split window based on number of components
        indexes.append(i * ns)  # starting index
        pu_.append([])
        for j in xrange(len(pulses)):
            if i == comp_num-1:
                pu_[-1].append(pulses[j][i*ns : ])
            else:
                pu_[-1].append(pulses[j][i*ns : (i+1)*ns])
    #print len(pu_)
    #print len(pu_[0])
    errfunc = lambda v, x, y: (f2(v, x) - y)

    for i in xrange(comp_num):
        for j in xrange(len(pu_[i])):
            bins = len(pu_[i][j])
            x_ = np.array(range(bins))
            v0 = [pu_[i][j][bins/3], bins/3., bins / 20., pu_[i][j][bins*2/3], bins*2./3., bins / 20.]
            ## Error function
            res = leastsq(errfunc, v0, args=(x_, pu_[i][j]), maxfev=1000, full_output=False)
            v = res[0]
            # check an offset
            mx = np.max(pu_[i][j])
            ind = list(pu_[i][j]).index(mx)
            #print i, j, v[1] - ind
            """
            #if np.fabs(v[1]-ind) > 3:
            ga0 = f2(v0, x_)
            ga = f2(v, x_)
            ga_1 = f(v[0:3], x_)
            ga_2 = f(v[3:], x_)
            import matplotlib.pyplot as pl
            xx_ = np.array(range(len(ga))) + indexes[i]
            pl.plot(pulses[j], c="blue")
            pl.plot(xx_, pu_[i][j], c="pink")
            pl.plot(xx_, ga, c="red")
            #pl.plot(xx_, ga0, c="green")  # init parameters
            pl.plot(xx_, ga_1, c="orange")
            pl.plot(xx_, ga_2, c="brown")
            pl.axvline(x=v[1]+indexes[i])
            #pl.show()
            pl.savefig("output/te.pdf")
            pl.close()
            a = raw_input()
            #"""
            peak = np.max(pu_[i])
            if v[0] > pthres * peak:
                max_x_[i].append(v[1] + indexes[i])
                max_y_[i].append(float(j))
            if v[3] >  pthres * peak:
                max_x_[i].append(v[4] + indexes[i])
                max_y_[i].append(float(j))
        #max_x_[i] += max_x2_[i]
        #max_y_[i] += max_y2_[i]
    return max_x_, max_y_


def get_maxima(pulses, comp_num, pthres=0.5, smooth=True):
    """
    Get maxima based on gaussian fitting
    """
    max_x_ = []
    max_y_ = []
    for i in xrange(comp_num):
        max_x_.append([])
        max_y_.append([])

    size = len(pulses[0])
    ns = size / comp_num
    pu_ = []
    indexes = []

    for i in xrange(comp_num):
        # split window based on number of components
        indexes.append(i * ns)  # starting index
        pu_.append([])
        for j in xrange(len(pulses)):
            if i == comp_num-1:
                pu_[-1].append(pulses[j][i*ns : ])
            else:
                pu_[-1].append(pulses[j][i*ns : (i+1)*ns])

    #print len(pu_)
    #print len(pu_[0])

    errfunc = lambda v, x, y: (f(v, x) - y)
    for i in xrange(comp_num):
        for j in xrange(len(pu_[i])):
            bins = len(pu_[i][j])
            x_ = np.array(range(bins))
            v0 = [pu_[i][j][bins/2], bins/2., bins / 20.]
            #print pu_[i][j][bins/2], bins/2., i, j
            ## Error function
            res = leastsq(errfunc, v0, args=(x_, pu_[i][j]), maxfev=1000, full_output=False)
            v = res[0]
            #print v0
            #print v
            # check an offset
            mx = np.max(pu_[i][j])
            ind = list(pu_[i][j]).index(mx)
            print i, j, v[1] - ind
            #"""
            if np.fabs(v[1]-ind) > 3:
                ga0 = f(v0, x_)
                ga = f(v, x_)
                import matplotlib.pyplot as pl
                xx_ = np.array(range(len(ga))) + indexes[i]
                pl.plot(pulses[j], c="blue")
                pl.plot(xx_, pu_[i][j], c="pink")
                pl.plot(xx_, ga, c="red")
                pl.plot(xx_, ga0, c="green")
                pl.axvline(x=v[1]+indexes[i])
                #pl.show()
                pl.savefig("output/te.pdf")
                pl.close()
                a = raw_input()
            #"""
            max_x_[i].append(v[1] + indexes[i])
            max_y_[i].append(float(j))
    return max_x_, max_y_


def get_p3_simple(signal, x, on_fail=0):
    base = pk.baseline(signal)
    signal -= base
    pind = pk.indexes(signal, min_dist=len(signal))
    if len(pind) > 0:
        freq = [x[pind[0]]]
        err = [0.005]  # not now
        p3 = 1. / freq[0]
        p3_err = p3 - (1. / (freq[0] + err[0]))
        p3_err2 = (1. / (freq[0] - err[0])) - p3
        return p3, np.max([p3_err, p3_err2]), pind[0]
    else:
        return None, None, None


def get_p3_old(signal, x, on_fail=0):
    base = pk.baseline(signal)
    signal -= base
    pind = pk.indexes(signal, min_dist=len(signal))
    try:
        freq, err = pk.interpolate2(x, signal, ind=pind, width=np.min([10, pind-2]), func=pk.gaussian_fit)
        print freq, err
    except:
        print 'Warning! Gaussian fit failed:'
        if on_fail == 0:
            print '\tmaximum used.'
            freq = [x[pind[0]]]
            err = [0.1]  # not now
        elif on_fail == 1:
            print '\tignored.'
            return None, None, None
    """
    print 1. / freq[0]
    nx = np.linspace(0, 0.1, num=500)
    ga = pk.gaussian(nx, 1., freq[0], err[0])
    from matplotlib import pyplot as pl
    pl.plot(x, signal)
    pl.plot(nx, ga, color="green")
    pl.axvline(x=freq[0], color="red")
    pl.axvline(x=x[pind], color="blue")
    pl.show()
    """
    p3 = 1. / freq[0]
    p3_err = p3 - (1. / (freq[0] + err[0]))
    p3_err2 = (1. / (freq[0] - err[0])) - p3
    #print np.max([p3_err, p3_err2])
    return p3, np.max([p3_err, p3_err2]), pind[0]


def f(v, x):
    res = v[0] * np.exp(-0.5*((x-v[1]) / v[2]) ** 2)
    return res

def f2(v, x):
    res = v[0] * np.exp(-0.5*((x-v[1]) / v[2]) ** 2) + v[3] * np.exp(-0.5*((x-v[4]) / v[5]) ** 2)
    return res


def get_p3(signal, x, thres=0.3):
    base = pk.baseline(signal)
    signal -= base
    # find peaks
    pind = pk.indexes(signal, thres=thres)
    freq_num = len(pind)

    if freq_num == 0:
        print "Warning! No maximum found: ignored"
        return None, None, None
    if freq_num == 1:
        ln = len(signal) - 1  # boundary case
        ind = pind[0]
        # crude width estimate
        mx = signal[ind]
        val = mx
        while val >= 0.5 * mx:
            ind += 1
            if ind == ln:  # boundary case
                break
            else:
                val = signal[ind]
        width = 2 * (ind - pind[0])  # 4.71 sigma
        #print "Width", width, pind[0]
        st = np.max([0, pind[0] - width])
        end = np.min([len(x), pind[0] + width])
        #print st, end, ind, width
        if pind[0] + width > ln:  # boundary case
            width = ln - pind[0]
        #print width, "\n"
        # gaussian fit
        v0 = [signal[pind[0]], x[pind[0]], (x[pind[0] + width] - x[pind[0]]) / 3.]
        ## Error function
        errfunc = lambda v, x, y: (f(v, x) - y)
        res = leastsq(errfunc, v0, args=(np.array(x[st:end]), np.array(signal[st:end])), maxfev=1000, full_output=False)
        v = res[0]
        #print v[1], freq[0]
        freq = v[1]
        err = v[2]
    elif freq_num >= 2:
        # crude widths estimate
        widths = []
        sts = []
        ends = []
        for i, index in enumerate(pind):
            mx = signal[index]
            val = mx
            ind = index
            while val >= 0.5 * mx:
                ind += 1
                try:
                    val = signal[ind]
                except IndexError:
                    ind = len(signal) - 1
                    break
            width = 2 * (ind - index)  # 4.71 sigma
            if index + width >= len(signal):
                width = len(signal) - index - 1
            if index - width < 0:
                width = index
            widths.append(width)
            #print "Width", width, pind[i]
            st = np.max([0, pind[i] - width])
            end = np.min([len(x), pind[i] + width])
            sts.append(st)
            ends.append(end)
        st = np.min(sts)
        end = np.max(ends)
        #print st, end, len(x), len(signal), widths[0], widths[1]
        v0 = [signal[pind[0]], x[pind[0]], (x[pind[0] + widths[0]] - x[pind[0]]) / 3., signal[pind[1]], x[pind[1]], (x[pind[1] + widths[1]] - x[pind[1]]) / 3.]
        ## Error function
        errfunc = lambda v, x, y: (f2(v, x) - y)
        try:
            res = leastsq(errfunc, v0, args=(np.array(x[st:end]), np.array(signal[st:end])), maxfev=1000, full_output=False)
        except TypeError:
            print "Warning! Gaussian fit error: ignored"
            return None, None, None
        v = res[0]
        if v[0] > v[3]:
            freq = v[1]
            err = v[2]
        else:
            freq = v[4]
            err = v[5]

        """
        xx = np.linspace(x[st], x[end], num=100)
        ga = f2(v0, xx)
        ga2 = f2(v, xx)
        from matplotlib import pyplot as pl
        pl.plot(x[st:end], signal[st:end])
        pl.plot(xx, ga)
        pl.plot(xx, ga2)
        pl.show()
        """
        print "Warning! Two or more maxima found: highest used..."
    #else:
    #    print "Warning! More than two maxima found: ignored"
    #    return None, None, None

    p3 = 1. / freq
    p3_err = p3 - (1. / (freq + err))
    p3_err2 = (1. / (freq - err)) - p3
    return p3, np.max([p3_err, p3_err2]), pind[0]



def get_p3_backup(signal, x, on_fail=0):

    base = pk.baseline(signal)
    signal -= base
    # find peaks
    pind = pk.indexes(signal)    #, min_dist=len(signal))  -> only one peak
    print len(pind)
    try:
        ind = pind[0]
    except:
        print "Warning! No maximum found: ignored"
        return None, None, None
    # crude width estimate
    mx = signal[ind]
    val = mx
    while val >= 0.5 *mx:
        ind += 1
        val = signal[ind]
    width = 2*(ind - pind[0])  # 4.71 sigma
    #print "Width", width, pind[0]
    st = np.max([0, pind[0]-width])
    end = np.min([len(x), pind[0]+width])

    #print "st", st, "end", end

    """
    # TODO start with this test!
    # frequency measurment
    freq = pk.interpolate(x, signal, ind=[pind[0]], width=np.min([width, pind[0]]), func=pk.gaussian_fit)
    # gaussian fit
    v0 = [signal[pind[0]], x[pind[0]], (x[pind[0] + width] - x[pind[0]]) / 3.]
    #y_new, v, errs = fun.least_sq_err(x[st:end], signal[st:end], f, v0, show=True)
    ## Error function
    errfunc = lambda v, x, y: (f(v, x) - y)
    res = leastsq(errfunc, v0, args=(np.array(x[st:end]), np.array(signal[st:end])), maxfev=1000, full_output=True)

    v = res[0]
    print res

    xx = np.linspace(x[st], x[end], num=100)
    ga = f(v0, xx)
    ga2 = f(v, xx)

    from matplotlib import pyplot as pl
    pl.plot(x[st:end], signal[st:end])
    pl.plot(xx, ga)
    pl.plot(xx, ga2)
    pl.show()

    print f(v0, x[pind[0]]), signal[pind[0]]
    print v
    print f(v, x[pind[0]]), signal[pind[0]]
    exit()
    """

    try:
        # frequency measurment
        freq = pk.interpolate(x, signal, ind=[pind[0]], width=np.min([width, pind[0]]), func=pk.gaussian_fit)
        # gaussian fit
        v0 = [signal[pind[0]], x[pind[0]], (x[pind[0] + width] - x[pind[0]]) / 3.]
        #y_new, v, errs = fun.least_sq_err(x[st:end], signal[st:end], f, v0, show=True)
        ## Error function
        errfunc = lambda v, x, y: (f(v, x) - y)
        res = leastsq(errfunc, v0, args=(np.array(x[st:end]), np.array(signal[st:end])), maxfev=1000, full_output=False)
        v = res[0]
        #print v[1], freq[0]
        freq = v[1]
        err = v[2]
        #params = pk.gaussian_fit(x[st:end], signal[st:end], center_only=False)
        #err = [params[2]]
    except:
        if on_fail == 0:
            print 'Warning! Gaussian fit failed: ignored!'
            return None, None, None
        else:
            print 'Warning! Gaussian fit failed: ignored!'
            return None, None, None  # not implemented yet

    p3 = 1. / freq
    """
    if p3 < 12:

        xx = np.linspace(x[st], x[end], num=100)
        ga = f(v0, xx)
        ga2 = f(v, xx)

        from matplotlib import pyplot as pl
        pl.plot(x[st:end], signal[st:end])
        pl.plot(xx, ga)
        pl.plot(xx, ga2)
        pl.show()

        nx = np.linspace(0, 0.5, num=500)
        ga = pk.gaussian(nx, v[0], v[1], v[2] )
        from matplotlib import pyplot as pl
        pl.close()
        pl.plot(x, signal)
        pl.plot(nx, ga, color="green")
        pl.axvline(x=freq[0], color="red")
        pl.axvline(x=x[pind[0]], color="blue")
        pl.show()
        pl.close()
    #"""
    p3_err = p3 - (1. / (freq + err))
    p3_err2 = (1. / (freq - err)) - p3
    #print np.max([p3_err, p3_err2])
    return p3, np.max([p3_err, p3_err2]), pind[0]


def get_p3_rahuls(signal, freq, thres=5., secs=5):
    sns = np.array_split(signal, secs)

    mean_ = np.zeros(secs)
    rms_ = np.zeros(secs)
    for i, sn in enumerate(sns):
        mean_[i] = np.mean(sn)
        rms_[i] = np.std(sn)

    #print mean_
    #print rms_
    rms = 1e50
    mean = 1e50
    for i in xrange(secs):
        if rms_[i] < rms:
            rms = rms_[i]
            mean = mean_[i]
    f_ = []
    v_ = []
    vtf_ = []
    inds = []
    for i, s in enumerate(signal):
        if s > mean + thres * rms:
            inds.append(i)
            f_.append(freq[i])
            v_.append(s)
            vtf_.append(s * freq[i])

    fp = np.sum(vtf_) / np.sum(v_)
    p3 = 1. / fp
    return p3, 0.01, 0


def fit_lines(xs_, ys_, rngs=None):
    """
    Fits lines to subpulse locations (for all components, with defined range - one for a component)
    """
    fun = lambda v, x: v[0] * x + v[1]
    x1s_ = []
    y1s_ = []
    vs = []
    es = []
    xs = []
    xes = []
    for i in xrange(len(xs_)):
        if rngs is None:
            x_ = xs_[i]
            y_ = ys_[i]
        else:
            x_ = xs_[i][rngs[i][0]:rngs[i][1]]
            y_ = ys_[i][rngs[i][0]:rngs[i][1]]
        x1_, y1_, v, err = least_sq_err(x_, y_, fun, [1., 1.], times_min=1.0, times_max=1.0)
        x1s_.append(x1_)
        y1s_.append(y1_)
        vs.append(v[0])
        es.append(err[0])
        # Note that you fit in different order! y, x!
        x_mean = np.mean(y_)
        x_max, x_min = np.max(y_), np.min(y_)
        xs.append(x_mean)
        xes.append(np.max([x_max-x_min, np.fabs(x_min-x_max)]))
    return np.array(x1s_), np.array(y1s_), vs, es, xs, xes


def fit_lineseq(x_, y_, rngs=None):
    """
    Fits lines to subpulse locations (for one component, with defined ranges - many ranges for a component)
    """
    fun = lambda v, x: v[0] * x + v[1]
    x1s_ = []
    y1s_ = []
    vs = []
    es = []
    xs = []
    xes = []
    if rngs is None:
        rngs = [(0, len(x_))]
    for i in xrange(len(rngs)):
        xx_ = x_[rngs[i][0]:rngs[i][1]]
        yy_ = y_[rngs[i][0]:rngs[i][1]]
        x1_, y1_, v, err = least_sq_err(xx_, yy_, fun, [1., 1.], times_min=1.0, times_max=1.0)
        x1s_.append(x1_)
        y1s_.append(y1_)
        vs.append(v[0])
        es.append(err[0])
        # Note that you fit in different order! y, x!
        x_mean = np.mean(yy_)
        x_max, x_min = np.max(yy_), np.min(yy_)
        xs.append(x_mean)
        xes.append(np.max([x_max-x_min, np.fabs(x_min-x_max)]))
    return np.array(x1s_), np.array(y1s_), vs, es, xs, xes


def fit_lines(xs_, ys_, rngs=None):
    """
    Fits lines to subpulse locations (for all components, with defined range - one for a component)
    """
    fun = lambda v, x: v[0] * x + v[1]
    x1s_ = []
    y1s_ = []
    vs = []
    es = []
    xs = []
    xes = []
    for i in xrange(len(xs_)):
        if rngs is None:
            x_ = xs_[i]
            y_ = ys_[i]
        else:
            x_ = xs_[i][rngs[i][0]:rngs[i][1]]
            y_ = ys_[i][rngs[i][0]:rngs[i][1]]
        x1_, y1_, v, err = least_sq_err(x_, y_, fun, [1., 1.], times_min=1.0, times_max=1.0)
        x1s_.append(x1_)
        y1s_.append(y1_)
        vs.append(v[0])
        es.append(err[0])
        # Note that you fit in different order! y, x!
        x_mean = np.mean(y_)
        x_max, x_min = np.max(y_), np.min(y_)
        xs.append(x_mean)
        xes.append(np.max([x_max-x_min, np.fabs(x_min-x_max)]))
    return np.array(x1s_), np.array(y1s_), vs, es, xs, xes





def single_pulses(pulses, start=0, end=None):
    """
    Select pulses to plot
    :param pulses: single pulses
    :param start: first pulse to use
    :param end: last pulse to use
    :return:
    """
    return pulses[start:end][:], start, end


def single_pulses_ratio(pulses, ratio=4./3., start=0, end=None):
    """
    Strange, but neat... Not necessary though, use aspect="auto" with imshow
    :param pulses: single pulses
    :param ratio: aspect ratio of resulting matrix (for imshow)
    :param start: first pulse to use
    :param end: Note that end pulse depends on ratio
    :return:
    """
    if end is not None:
        pulses = pulses[start:end][:]
    else:
        pulses = pulses[start:][:]

    yl, xl = pulses.shape
    ma = max([yl, xl])
    # new sizes
    xn = ma
    yn = int(ma / ratio)
    # modifications
    xm = float(xn) / xl
    ym = yn / yl + 1.
    profile_ = np.zeros([yn, xn])
    for i in xrange(yn):
        for j in xrange(xn):
            profile_[i][j] = pulses[int(i/ym)][int(j/xm)]
    y_max = start + int(i / ym)
    y_min = start
    return profile_, y_min, y_max


def check_spdistance(x, y, radius, center):
    """
    checks if spark is at the polar cap
    :param x: spark x-location
    :param y: spark y-location
    :param radius: radius of the polar cap
    :param center: location of the polar cap center
    :return: distance from the center / polar cap radius
    """
    dr = np.sqrt((x - center[0]) ** 2. + (y - center[1]) ** 2.)
    return dr / radius


def get_theta_old(r, x, y, r_x, r_y):
    """
    GO HOME!
    get an angle of a spark for circular motion around the rotation axis
    :param r: distance from rotation axis
    :param x: location of a spark
    :param y: location of a spark
    :param r_x: location of rotation axis
    :param r_y: location of rotation axis
    :return:
    """
    theta = sy.symbols("theta")
    res = sy.solve([r * sy.cos(theta) - x + r_x, r * sy.sin(theta) - y + r_y], theta, dict=True)
    return res[0]["theta"]


def get_theta(r, x, y, r_x, r_y):
    """
    get an angle of a spark for circular motion around the rotation axis
    :param r: distance from rotation axis
    :param x: location of a spark
    :param y: location of a spark
    :param r_x: location of rotation axis
    :param r_y: location of rotation axis
    :return:
    """
    #print np.arccos((x - r_x) / r), np.arcsin((y - r_y) / r), np.arctan(y/x-r_y/r_x), np.arctan2(y-r_y, x-r_x)
    return np.arctan2(y-r_y, x-r_x)

def average_profile_soft(prof_):
    prof_ = np.array(prof_)
    #yl, xl = prof_.shape
    yl = len(prof_)
    min_ = len(prof_[0])
    for i in xrange(yl):
        if len(prof_[i]) < min_:
            min_ = len(prof_[i])
    xl = min_
    ave_ = np.zeros(xl)
    for i in xrange(yl):
        for j in xrange(xl):
            ave_[j] += prof_[i][j]
    return ave_


def average_profile(prof_):
    prof_ = np.array(prof_)
    yl, xl = prof_.shape
    ave_ = np.zeros(xl)
    for i in xrange(yl):
        for j in xrange(xl):
            ave_[j] += prof_[i][j]
    return ave_


def counts(prof_):
    prof_ = np.array(prof_)
    yl, xl = prof_.shape
    counts_ = np.zeros(yl)
    pulses_ = np.zeros(yl)
    for i in xrange(yl):
        counts_[i] = sum(prof_[i])
        pulses_[i] = i
    mx = np.max(counts_)
    counts_ /= mx
    return counts_, pulses_


def lrfs(single_, psr):
    # dp = psr.p  # sample spacing NO NO NO (freq in 1/P use 1.)
    single_ = np.array(single_)  # i - pulse number, j - bin number
    new_ = single_.transpose()
    xl, yl = new_.shape  # i - bin number, j - pulse number
    freq = np.fft.fftfreq(yl, d=1.)[1:yl/2]  # one side frequency range
    ffts = []
    for i in xrange(xl):
        fft = np.fft.fft(new_[i])  # fft computing for a specific bin/longitude, normalization? (/ yl)
        ffts.append(fft[1:yl/2])  # for one side frequency range
    ffts = np.array(ffts)
    ffts = ffts.transpose()  # for imshow
    return ffts, freq


def lrfs_phase(max_ind, lrfs_):
    ffph_ = to_degrees(np.angle(lrfs_[max_ind]))
    """
    for i in xrange(len(ffph_)):
        if ffph_[i] < 0.:
            ffph_[i] += 360.
        # HACK!
        if ffph_[i] < 100.:
            ffph_[i] += 360.
    """
    return ffph_


def add_phase(phase_, single_, dphase=10., noise_level=0.):
    """
    adds phase
    :param phase_: phase array
    :param single_: single_pulse array (2D)
    :param dphase: phase to add in deg
    :return:
    """
    dp = phase_[1] - phase_[0]

    di = int(dphase / dp)

    """
    # it really worked? wow! check the next line and take a break!
    new_phase_ = np.zeros(di*2 + len(phase_))
    for i in xrange(di):
        new_phase_[i] = phase_[0] - (di-i) * dp
    for i in xrange(di, di+len(phase_)):
        new_phase_[i] = phase_[i-di]
    for i in xrange(di+len(phase_), 2*di+len(phase_)):
        new_phase_[i] = phase_[-1] + (i+1-(di+len(phase_))) * dp
    """
    new_phase_ = np.linspace(phase_[0]-dphase, phase_[-1]+dphase, len(phase_)+2*di)
    if noise_level != 0.:
        new_single_ = np.random.rand(len(single_), len(single_[0])+2*di) * noise_level - noise_level / 2.
    else:
        new_single_ = np.zeros([len(single_), len(single_[0])+2*di])

    for i in xrange(len(single_)):
        for j in xrange(di, di+len(single_[i])):
            new_single_[i][j] = single_[i][j-di]

    return new_phase_, new_single_

def fold_single(single_, p3, ybins=10, start_ind=None):
    """
    creates folded profile
    :param single_: single pulses
    :param p3: P_3 periodicity
    :return:
    """
    if start_ind is None:
        start_ind = int(p3 / 2.)

    single_ = single_[start_ind:]

    folded_ = np.zeros([ybins, len(single_[0])])

    dp3 = p3 / float(ybins)  # p3 step for ybin

    for i in xrange(len(single_)):
        new_ind = i / dp3
        j = int(modd(new_ind, 0, ybins))
        folded_[j] += single_[i]
    return folded_

def modd(val, min_, max_):
    dv = max_ - min_
    while val < min_:
        val += dv
    while val >= max_:
        val -= dv
    return val


def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    r"""Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techniques.
    Parameters
    ----------
    y : array_like, shape (N,)
        the values of the time history of the signal.
    window_size : int
        the length of the window. Must be an odd integer number.
    order : int
        the order of the polynomial used in the filtering.
        Must be less then `window_size` - 1.
    deriv: int
        the order of the derivative to compute (default = 0 means only smoothing)
    Returns
    -------
    ys : ndarray, shape (N)
        the smoothed signal (or it's n-th derivative).
    Notes
    -----
    The Savitzky-Golay is a type of low-pass filter, particularly
    suited for smoothing noisy data. The main idea behind this
    approach is to make for each point a least-square fit with a
    polynomial of high order over a odd-sized window centered at
    the point.
    Examples
    --------
    t = np.linspace(-4, 4, 500)
    y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
    ysg = savitzky_golay(y, window_size=31, order=4)
    import matplotlib.pyplot as plt
    plt.plot(t, y, label='Noisy signal')
    plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
    plt.plot(t, ysg, 'r', label='Filtered signal')
    plt.legend()
    plt.show()
    References
    ----------
    .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
       Data by Simplified Least Squares Procedures. Analytical
       Chemistry, 1964, 36 (8), pp 1627-1639.
    .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
       W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
       Cambridge University Press ISBN-13: 9780521880688
    """
    #import numpy as np
    #from math import factorial

    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except ValueError, msg:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs(y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve(m[::-1], y, mode='valid')


def zeroed(phase_):
    """
    shift phase to have zero latitude in the middle
    :param phase_: phase array
    :return:
    """
    phase_ = np.array(phase_)
    ph_min = phase_[0]
    ph_max = phase_[-1]
    dph = ph_max - ph_min
    phase_ -= (ph_min + dph / 2.)
    return phase_


