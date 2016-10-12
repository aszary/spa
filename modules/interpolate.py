from scipy.optimize import leastsq, curve_fit
import numpy as np
import scipy.odr as sciodr


def least_sq(x, y, fun, v0, size=200, times_min=0.9, times_max=1.1):
    """
    no errors
    """
    x_0 = min(x)
    x_1 = max(x)
    ## Error function
    errfunc = lambda v, x, y: (fun(v, x) - y)
    res = leastsq(errfunc, v0, args=(np.array(x), np.array(y)), maxfev=10000, full_output=True)
    v, conv = res[0], res[1]
    x_new = np.linspace(times_min*x_0, times_max*x_1, size)
    y_new = fun(v, x_new)
    return x_new, y_new, v

def least_sq_err(x, y, fun, v0, size=200, times_min=0.9, times_max=1.1, show=False):
    """
    with errors
    """
    x_0 = min(x)
    x_1 = max(x)
    ## Error function
    errfunc = lambda v, x, y: (fun(v, x) - y)
    res = leastsq(errfunc, v0, args=(np.array(x), np.array(y)), maxfev=10000, full_output=True)
    v, conv = res[0], res[1]
    chi_sq_red2 = (res[2]['fvec'] ** 2.).sum() / (len(y)-len(v))
    res_errs = conv * chi_sq_red2
    errs = (np.absolute(res_errs[0][0])**0.5, np.absolute(res_errs[1][1])**0.5)
    if show is True:
        print 'convolution (jacobian around the solution)', conv
        print 'chi square:', sum(pow(errfunc(v, np.array(x), np.array(y)), 2.))
        print 'chi^2_red = ', chi_sq_red2
        print 'Parameters:', v
        print 'Errors:', errs
        #print sum(pow(errfunc(v, np.array(x), np.array(y)), 2.))
    x_new = np.linspace(times_min*x_0, times_max*x_1, size)
    y_new = fun(v, x_new)
    return x_new, y_new, v, errs


def least_sq1D(x, y, fun, err, v0, size=200, times_min=0.9, times_max=1.1):
    """
        err is 1D array (len=1)
    """
    x_0 = min(x)
    x_1 = max(x)
    ## Error function
    errfunc = lambda v, x, y, err: (fun(v, x) - y) / err
    res = leastsq(errfunc, v0, args=(np.array(x), np.array(y), np.array(err)), maxfev=10000, full_output=True)
    v, conv = res[0], res[1]
    print 'convolution (jacobian around the solution)', conv
    print 'chi square:', sum(pow(errfunc(v, np.array(x), np.array(y), np.array(err)), 2.))
    chi_sq_red2 = (res[2]['fvec'] ** 2.).sum() / (len(y)-len(v))
    res_errs = conv * chi_sq_red2
    errs = (np.absolute(res_errs[0][0])**0.5, np.absolute(res_errs[1][1])**0.5)
    print 'chi^2_red = ', chi_sq_red2
    print 'Parameters:', v
    print 'Errors:', errs
    x_new = np.linspace(times_min*x_0, times_max*x_1, size)
    y_new = fun(v, x_new)
    return x_new, y_new, v, errs

def least_sq2D(x, y, fun, err, v0, size=200, times_min=0.9, times_max=1.1):
    """
        err is 2D array (len=2) - minus error [0], plus error [1]
    """
    x_0 = min(x)
    x_1 = max(x)
    x = np.array(x)
    y = np.array(y)
    ## Error function
    def errfunc(v, x, y, err):
        diff = fun(v, x) - y
        # err plus or minus?
        for i in xrange(len(diff)):
            if diff[i] < 0:
                diff[i] /= err[0][i]
            else:
                diff[i] /= err[1][i]
        return diff
    errors = np.array([np.array(err[0]), np.array(err[1])])
    res = leastsq(errfunc, v0, args=(x, y, errors), maxfev=10000, full_output=True)
    v, conv = res[0], res[1]
    #chi_sq_red = (errfunc(v, x, y, errors) ** 2.).sum() / (len(y)-len(v))
    chi_sq_red2 = (res[2]['fvec'] ** 2.).sum() / (len(y)-len(v))
    res_errs = conv * chi_sq_red2
    errs = (np.absolute(res_errs[0][0])**0.5, np.absolute(res_errs[1][1])**0.5)
    print 'chi^2_red = ', chi_sq_red2
    print 'Parameters:', v
    print 'Errors:', errs
    #print sum(pow(errfunc(v, np.array(x), np.array(y), errors), 2.))
    x_new = np.linspace(times_min*x_0, times_max*x_1, size)
    y_new = fun(v, x_new)
    #y_new = fun([v[0]+res_errs[0][1], v[1]], x_new)
    return x_new, y_new, v, errs

def odr(x, y, x_err, y_err, func, params, times_min=0.9, times_max=1.1, size=200):
    """
    Orthogonal distance regression http://docs.scipy.org/doc/scipy/reference/odr.html
    Alternatives: http://stackoverflow.com/questions/22670057/linear-fitting-in-python-with-uncertainty-in-both-x-and-y-coordinates

    # Define a function (quadratic in our case) to fit the data with.
    def quad_func(p, x):
        m, c = p
        return m*x**2 + c
    """
    model = sciodr.Model(func)
    data = sciodr.RealData(x, y, sx=x_err, sy=y_err)
    odr = sciodr.ODR(data, model, beta0=params, maxit=10000)
    # Run the regression.
    out = odr.run()
    out.pprint()
    x_0 = min(x)
    x_1 = max(x)
    x_new = np.linspace(times_min * x_0, times_max * x_1, size)
    y_new = func(out.beta, x_new)
    return x_new, y_new, out.beta, out.sd_beta



'''
def curve_fit2D(x, y, fun, err, v0, size=200, times_min=0.9, times_max=1.1):
    """
        err is 2D array (len=2) - minus error [0], plus error [1]
    """
    x_0 = min(x)
    x_1 = max(x)
    ## Error function
    def errfunc(v, x, y, err):
        diff = fun(v, x) - y
        # err plus or minus?
        for i in xrange(len(diff)):
            if diff[i] < 0:
                diff[i] /= err[0][i]
            else:
                diff[i] /= err[1][i]
        return diff
    errors = np.array([np.array(err[0]), np.array(err[1])])
    res = curve_fit(errfunc, x, y, rgs=(np.array(x), np.array(y), errors), maxfev=10000, full_output=True)
    v, success = res[0], res[1]
    print res
    x_new = np.linspace(times_min*x_0, times_max*x_1, size)
    y_new = fun(v, x_new)
    return x_new, y_new, v
'''