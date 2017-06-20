# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 15:16:42 2017

@author: Connor
"""

import numpy as np
import scipy.integrate as itg
import matplotlib.pyplot as plt
import tc_func as tf


def A(zeta_m, D):
    return 2/np.pi*np.arctan((D/2)/zeta_m)


def integrand(e, zeta_m):
        return (1/np.pi)/(zeta_m**2 + e**2)


def init_summand(w_n, n, w_m, w_e, D):
#    return tf.lam_odd(w_e, w_m, w_n)*w_n*itg.quad(
#            integrand, -D/2, D/2, args=(w_n), limit=100)[0]
    return tf.lam_odd(w_e, w_m, w_n)*A(w_n, D)


def summand(w_n, n, w_m, zeta, w_e, t, D):
#    return tf.lam_odd(w_e, w_m, w_n)*zeta[
#                np.int(n-1)]*itg.quad(
#        integrand, -D/2, D/2, args=(zeta[
#                np.int(n-1)]), limit=100)[0]
    return tf.lam_odd(w_e, w_m, w_n)*A(zeta[n-1], D)


def zeta_solver(t, g, w_e, dom_lim, D, maxiter=150,
                tol=1e-3, iprint=False, diff=False, damp=0.3):
    # n takes the place of m'

    llam = 2*g**2/(D*w_e)

#    print('*********************\ng: %g, w_e: %g, D: %g' % (g, w_e, D))
#    print('lambda = %g' % llam)
    """
    attempts to converge the zeta function, from an initial guess
    where zeta(iw_m)=w_m.
    A damping factor has been introduced to aid in convergence

    Input: Tc, g, w_e, limits of frequency indices to be calc'd over,
    limits of Matsu sum, max Ncber of iterations, damping factor

    Output: a callable function zeta(x), converged for the given Tc


    index i of new_zeta is equal to new_zeta evaluated at m=i+1
    """
    Nc = dom_lim + 30

    if iprint:
        plt.figure()
        plt.grid(True)
        plt.plot(tf.m_array(1, dom_lim), tf.freq_array(
                1, dom_lim, t), '.', markersize='2')

    diff_vec = []
    new_zeta = np.zeros(Nc)
    # constant DOS, using band of -D/2,D/2
    # for first loop, use zeta that is equal to w_m
    for m in tf.m_array(1, Nc):
        w_m = tf.freq_m(m, t)
        new_zeta[m-1] = w_m + llam*t*np.pi*tf.matsu_sum(1, Nc,
                                                        t, init_summand,
                                                        w_m, w_e, D)

    if iprint:
        plt.plot(tf.m_array(1, dom_lim),
                 new_zeta[:dom_lim], '--', label='it 0')

    diff_vec.append(tf.f_compare(tf.freq_array(1, Nc, t), new_zeta))

    """
    we now have a zeta function for the RHS, so we continue to iterate
    until the difference becomes small
    """

    for i in range(1, maxiter+1):
        old_zeta = np.copy(new_zeta)
        for m in tf.m_array(1, Nc):
            w_m = tf.freq_m(m, t)
#            new_zeta[m-1] = w_m
            new_zeta[m-1] = (1-damp)*(
                    w_m + llam*t*np.pi*tf.matsu_sum(1, Nc, t, summand,
                                                    w_m, old_zeta, w_e, t, D
                                                    )) + damp*old_zeta[m-1]
        diff_vec.append(tf.f_compare(old_zeta, new_zeta))

        if(np.mod(i, maxiter // 5) == 0):
            if iprint:
                print('Difference in iteration %i is ' % i, diff_vec[i])
                plt.plot(tf.m_array(1, dom_lim), new_zeta[:dom_lim], '-.',
                         label='it %i' % i)
        if (diff_vec[i] < tol):
            if iprint:
                print('zeta converged to tol in %i iterations' % i)
                plt.plot(tf.m_array(1, dom_lim), new_zeta[:dom_lim], '-.',
                         label='it %i' % i)
            break
        if i == maxiter:
            print('Zeta did not converge in %i iterations' % i,
                  'damp = %2.1f' % damp)
            print('last difference: ', diff_vec[-1])

    if iprint:
        print('last difference: ', diff_vec[-1])
        plt.legend(loc='best')
        plt.title('Zeta fnc. Damping = %2.2f' % damp)
        plt.savefig('zeta_func.pdf', bbox_inches='tight')
        plt.ylabel('Zeta(m)')
        plt.xlabel('m')
        plt.show()

        plt.figure()
        plt.plot(np.log(diff_vec), 'o', markersize=2)
        plt.xlabel('Iteration n')
        plt.ylabel('Log Diff')
        plt.title('Log difference in iterated fnc. Damping =%2.2f' % damp)
        plt.show()
    zeta = tf.interpolater(tf.freq_array(1, Nc, t), new_zeta)
    if diff:
        return zeta, diff_vec,
    else:
        return zeta
