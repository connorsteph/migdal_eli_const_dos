# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 11:44:05 2017

@author: Connor
"""

import numpy as np
from scipy.integrate import quad
from matplotlib import pyplot as plt
from scipy.optimize import brentq
import tc_func as tf
from zeta_solver import zeta_solver


def A(zeta_m, D):
    return 2/np.pi*np.arctan(D/2/zeta_m)


def tc_root_eqn(
        t, g, w_e, D, phi, dom_lim,
        maxiter=30, damp=0.3):
    Nc = 75
    zeta = zeta_solver(t, g, w_e, Nc, D)
    llam = 2*g**2/(D*w_e)
    try:
            return np.pi*llam*t*tf.matsu_sum(1, Nc, t, init_summand,
                                             tf.freq_m(1, t), phi, zeta, w_e,
                                             t, D) - phi(tf.freq_m(1, t))
    except TypeError:
            return np.pi*llam*t*tf.matsu_sum(1, Nc, t, init_summand,
                                             tf.freq_m(1, t), phi, zeta,
                                             w_e, t, D) - phi(1, t)


def integrand(e, zeta_n,):
    return (1/np.pi)/(zeta_n**2 + e**2)


def init_summand(w_n, n, w_m, phi, zeta, w_e, t, D):
#    try:
#        return tf.lam_even(w_e, w_m, w_n)*phi(w_n)*quad(
#                integrand, -D/2, D/2, args=(zeta(w_n),), limit=100)[0]
#    except TypeError:
#        return tf.lam_even(w_e, w_m, w_n)*phi(n, t)*quad(
#                integrand, -D/2, D/2, args=(zeta(w_n),), limit=100)[0]
    try:
        return tf.lam_even(w_e, w_m, w_n)*phi(w_n)/zeta(w_n)*A(zeta(w_n), D)
    except TypeError:
        return tf.lam_even(w_e, w_m, w_n)*phi(n, t)/zeta(w_n)*A(zeta(w_n), D)


def summand(w_n, n, w_m, phi, zeta, w_e, t, D):
#    return tf.lam_even(
#            w_e, w_m, w_n)*phi[n-1]*quad(
#        integrand, -D/2, D/2, args=(zeta(w_n),), limit=100)[0]
    return tf.lam_even(w_e, w_m, w_n)*phi[n-1]/zeta(w_n)*A(zeta(w_n), D)


def phi_solver(g, w_e, dom_lim, D, init_phi, maxiter=100, p_damp=0.3,
               iprint=False, tol=1e-5, damp=0.3, p_tol=1e-2,
               t_tol=5e-2, tc=None,
               ):
    l_root = 0.01*w_e
    llam = 2*g**2/(D*w_e)

#    print('*********************\ng: %g, w_e: %g, D: %g' % (g, w_e, D))
#    print('lambda = %g\n*********************' % llam)
#
#    print('\nphi solver\n\n*********************')

    diff_vec = []
    Nc = dom_lim + 25
    new_phi = np.zeros(Nc)

    if tc is None:
        if iprint:
            print('Solving for initial tc')
            plt.figure()
            plt.grid(True)
            num = 11
            t_domain = np.linspace(l_root, w_e, num)
            y = np.zeros(num)
            for c, t in enumerate(t_domain, 0):
                y[c] = (tc_root_eqn(t, g, w_e, D, init_phi, dom_lim))
            plt.plot(t_domain, y, 'o-')
            plt.xlim([0, w_e])
            plt.xlabel('t')
            plt.show()

        tc = brentq(tc_root_eqn, l_root, w_e, args=(
                g, w_e, D, init_phi, dom_lim))

#    print('Tc/w_e = %5.4g' % (tc/w_e))
#    print('Converging zeta')
    zeta = zeta_solver(tc, g, w_e, Nc, D, tol=tol, iprint=False)

    if iprint:
        plt.figure()
        plt.grid(True)
        plt.ylim([0, 1])
        plt.xlabel('w_m')
        try:
            plt.plot([w/w_e for w in tf.freq_array(1, dom_lim, tc)],
                     [init_phi(w)/init_phi(tf.freq_m(1, tc))
                     for w in tf.freq_array(1, dom_lim, tc)],
                     label='initial')
        except TypeError:
            plt.plot([w/w_e for w in tf.freq_array(1, dom_lim, tc)],
                     [init_phi(w, tc)/init_phi(tf.freq_m(1, tc), tc)
                     for w in tf.freq_array(1, dom_lim, tc)],
                     label='initial')

#    print('iterating initial phi')
    for m in tf.m_array(1, Nc):
        w_m = tf.freq_m(m, tc)
        new_phi[m-1] = llam*tc*np.pi*tf.matsu_sum(
                    1, Nc, tc, init_summand,
                    w_m, init_phi, zeta, w_e, tc, D)
        new_phi = np.copy([new_phi[i]/new_phi[0]
                          for i in range(Nc)])

#    print('converging phi')
    for i in range(1, maxiter+1):
        old_phi = np.copy(new_phi)
        for m in tf.m_array(1, Nc):
            w_m = tf.freq_m(m, tc)
            new_phi[m-1] = (1-p_damp)*llam*tc*np.pi*tf.matsu_sum(
                                1, Nc, tc,
                                summand, w_m, old_phi, zeta, w_e, tc, D
                                )+p_damp*old_phi[m-1]
        new_phi = np.copy([new_phi[i]/new_phi[0] for i in range(Nc)])
        diff_vec.append(tf.f_compare(old_phi, new_phi))

        if iprint:
            if(np.mod(i, maxiter // 5) == 0):
                print('Difference in iteration %i is ' % i, diff_vec[i-1])
                print('pdamp = %g' % p_damp)
                plt.plot([w/w_e for w in tf.freq_array(1, dom_lim, tc)],
                         new_phi[:dom_lim], '-.', label='it %i' % i)
        if (diff_vec[i-1] < p_tol):
            if iprint:
                print('phi converged to p_tol in %i iterations' % i)
                print('p_tol = %3.2g, Nc = %i, g =%3.2g, w_e =%3.2g,\n\
                D =%3.2g, p_damp = %3.2g'
                      % (p_tol, Nc, g, w_e, D, p_damp))
                plt.plot([w/w_e for w in tf.freq_array(1, dom_lim, tc)],
                         new_phi[:dom_lim], '-.', label='it %i' % i)
            break

    if iprint:
        plt.legend(loc='best')
        plt.savefig('phi_iter.pdf', bbox_inches='tight')
        plt.show()

        plt.figure()
        plt.plot(np.log(diff_vec), 'o', markersize=2)
        plt.xlabel('Iteration n')
        plt.ylabel('Log Diff')
        plt.title('Log difference in phi iter. Damping = %2.2f' % p_damp)
        plt.show()

    phi = tf.interpolater(tf.freq_array(1, dom_lim, tc), new_phi[:dom_lim])
    try:
        old_vals = [init_phi(w)/init_phi(tf.freq_m(1, tc))
                    for w in tf.freq_array(1, dom_lim, tc)]
    except TypeError:
        old_vals = [init_phi(tf.matsu_index(w, tc), tc)/init_phi(1, tc)
                    for w in tf.freq_array(1, dom_lim, tc)]

    diff = tf.f_compare(old_vals, new_phi[:dom_lim])
#    print("Diff = %4.3g" % diff)

#    print('plotting new tc roots')
#    plt.figure()
#    plt.grid(True)
#    num = 11
#    t_domain = np.linspace(l_root, w_e, num)
#    y = np.zeros(num)
#    for c, t in enumerate(t_domain, 0):
#        y[c] = (tc_root_eqn(t, g, w_e, D, phi, dom_lim))
#    plt.plot(t_domain, y, 'o-')
#    plt.xlabel('t')
#    plt.xlim([0, w_e])
#    plt.show()

    new_tc = brentq(tc_root_eqn, l_root, w_e, args=(
            g, w_e, D, phi, dom_lim))
    t_diff = abs(tc - new_tc)
#    print("New tc/w_e: %5.4g, t_diff = %5.4g" % ((new_tc/w_e), (t_diff/w_e)))
#    print('w_e: ', w_e)
    dom_lim = np.int(tc/new_tc*(dom_lim-1/2)+1/2)
#    print(dom_lim)
    # keeps new dom_lim from extending past domain of the initial phi
    # means that the initial dom_lim needs to be larger to compensate
    # if tc increases
    if (diff < p_tol and t_diff < t_tol):
        return phi, new_tc, dom_lim
    else:
        return phi_solver(g, w_e, dom_lim, D, phi,
                          maxiter=maxiter, p_damp=p_damp,
                          iprint=iprint, tol=tol, tc=new_tc
                          )
