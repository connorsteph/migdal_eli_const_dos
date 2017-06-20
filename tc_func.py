# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 14:59:33 2017

@author: Connor
"""
import numpy as np
import scipy.integrate as integral

"""
important to have a data file with ttx and ttp data in the two column format
"""

data = np.loadtxt('ttx_0d0_nx_500_dee_1d-2.dat', skiprows=1, usecols=[0, 1])
dos_vals = data[1:, :]
[ttp, ttx] = data[0, :]

e_min = -8/ttp-6*abs(ttx)/ttp
e_max = 8/ttp+6*abs(ttx)/ttp


def interpolater(f_domain, f_range):
    """
    given a set of x_i, f(x_i), returns an interpolated function
    f(x) defined over x in {x_i_min -> x_i_max}
    """
    def func(x):
        if isinstance(x, (list, tuple, np.ndarray)):
            vals = []
            for i in x:
                l_index = binsearch(f_domain, i)
                if l_index+1 == np.size(f_domain):
                    return
                vals.append(f_range[l_index]
                            + (i-f_domain[l_index])*(f_range[l_index+1]
                            - f_domain[l_index])/(f_range[l_index+1]
                            - f_range[l_index]))
            return vals
        else:
            l_index = binsearch(f_domain,x)
            return (f_range[l_index]
                    +(x-f_domain[l_index])*(f_range[l_index+1]
                    -f_range[l_index])/(f_domain[l_index+1]-f_domain[l_index]))
    return func


def binsearch(vec, val):
    """
    returns lower bracketing index for the closest value in vec to val
    """
    m = np.size(vec)
    ia = 0
    ib = m-1

    while (ib-ia) > 1:
        im = (ia + ib)//2
        if vec[im] < val:
            ia = im
        else:
            ib = im
    return ia


def ek(z, y, x, rho):
    # bcc NNN dispersion
    return -8*(np.cos(x)*np.cos(y)*np.cos(z))
    -rho*2*(np.cos(x) + np.cos(y) + np.cos(z))


def f(x, t):
    # fermi distr.
    return 1/(1+np.exp(x/t))


def n(x, t):
    # bose einstein distr.
    return 1/(np.exp(x/t) - 1)


def dos(e):
    # interpolates the DOS from a given file
    if isinstance(e, (list, tuple, np.ndarray)):
        vals = []
        for i in e:
            l_index = binsearch(dos_vals[:, 0], i)
            if l_index+1 == dos_vals[:, 0].size:
                return
            vals.append(dos_vals[l_index, 1]
                        + (i-dos_vals[l_index, 0])*(dos_vals[l_index+1, 1]
                        - dos_vals[l_index, 1])/(dos_vals[l_index+1, 0]
                        - dos_vals[l_index, 0]))
        return vals
    else:
        l_index = binsearch(dos_vals[:, 0], e)
        return (dos_vals[l_index,1]
                + (e-dos_vals[l_index,0])*(dos_vals[l_index+1,1]
                - dos_vals[l_index, 1])/(dos_vals[l_index + 1, 0]
                - dos_vals[l_index, 0]))


def s0(e, w, t, w_e, mu):
    """
    s0 function to be integrated
    """
    return (1/2*((n(w_e) + 1 - f(e - mu))/(w - w_e - (e - mu))
                 + (n(w_e) + f(e - mu))/(w + w_e - (e - mu))))


def freq_m(n, t):
    """
    returns the fermion matsubara frequency corresponding to integer n
    """
    return np.pi*t*(2*n-1)


def matsu_index(w, t):
    """
    returns the integer n which gives the fermion Matsubara freq.
    w_n=pi*t(2*n-1)
    """
    return round(1/2*(w/np.pi/t+1))


def matsu_sum(lower, upper, t, f, *args):
    ssum = 0.0
    """
    returns the sum over fermion matsubara freq. (m = lower to upper)
    of func. f, with arguments given by
    **kwargs, at temperature t
    """
    for n in range(lower, upper+1, 1):
        ssum += f(np.pi*t*(2*n-1), n, *args)
    return ssum


def matsu_sym(lim, t, f, *args):
    """
    returns the symmetric sum over fermion matsubara freq.
    (from m = 1 to lim) of func. f,
    with arguments given by **kwargs, at temperature t
    """

    sum_range = [f(w, *args) for w in freq_array(-lim, lim, t)]
    return sum(sum_range)


def freq_array(lower, upper, t):
    """
    returns matsubara frequencies m= lower to upper
    """
    return [freq_m(n, t) for n in m_array(lower, upper)]


def m_array(lower, upper):
    """
    returns corresponding m for f_range
    """

    return range(lower, upper+1, 1)


def sig0(lower, upper, t, w_e, mu):
    """
    Samples sig0 over a range of wm, with m from lower to upper, at temp t.
    """
    return [integral.quad(s0, e_min, e_max, args=(w, t, w_e, mu))[0]
            for w in freq_array(lower, upper, t)]


def f_compare(v1, v2):
    """
    returns the average absolute value of the difference between
    two vector's components
    """
    diff = 0.0
    assert(np.size(v1) == np.size(v2))
    for i in range(np.size(v1)):
        diff += abs((v1[i]-v2[i]))
    return diff/np.size(v1)


def lam_even(w_e, w_m, w_n):
    """
    even phonon kernel with imaginary freq. arguments, without
    mass enhancement param. (folded)
    """
    return w_e**2*(1/(w_e**2+(w_m-w_n)**2)+1/(w_e**2+(w_m+w_n)**2))


def lam_odd(w_e, w_m, w_n):
    """
    odd phonon kernel with imaginary freq. arguments, without
    mass enhancement param. (folded)
    """
    return w_e**2*(1/(w_e**2+(w_m-w_n)**2)-1/(w_e**2+(w_m+w_n)**2))


def lam_full(w_e, w_m, w_n):
    return w_e**2/(w_e**2+(w_m-w_n)**2)
