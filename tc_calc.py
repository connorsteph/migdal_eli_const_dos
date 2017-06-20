# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 15:08:35 2017

@author: Connor
"""

import numpy as np
from matplotlib import pyplot as plt
from time import time
import tc_func as tf
from zeta_solver import zeta_solver
from phi_solver import phi_solver
from phi_solver import init_summand
"""
Pseudo Code:
*************************

1. begin with a guess for Tc, and then converge zeta.

2. using an initial guess for phi, calculate Tc by
    rootfinding an equation for some phi_n

3. using this new Tc, iterate phi to convergence (starting with our guess)

4. repeat 2 using the new guess for phi

5. repeat steps 3,4

*************************

"""


def init_phi(n, t):
    return 1/tf.freq_m(n, t)


def tc_calc(g, w_e, D, dom_lim, iprint=False, tol=1e-8, p_damp=0.3,
            maxiter=150, t_tol=5e-2, p_tol=1e-2, plot=False):
    start = time()
    phi, tc, new_dom_lim = phi_solver(g, w_e, dom_lim, D,
                                      init_phi, maxiter=maxiter, p_damp=p_damp,
                                      iprint=iprint, tol=tol, p_tol=p_tol,
                                      t_tol=t_tol
                                      )
    end = time()
    print('Runtime: ', end-start)
    if plot:
        llam = 2*g**2/(D*w_e)
        plt.figure(num=None, figsize=(6, 8), dpi=150,
                   facecolor='w', edgecolor='k')
        plt.grid(True)
        plt.ylim([0, 1])
        domain = [w/w_e for w in tf.freq_array(1, dom_lim, tc)]
        plt.plot(domain,
                 [phi(w) for w in tf.freq_array(1, dom_lim, tc)])
        plt.ylim([-1, 1])
        plt.xlim([0, tf.freq_m(dom_lim, tc)/w_e])
        plt.xlabel(r'$\frac{\omega_m}{\omega_E}$', fontsize=18)
        plt.title(r'$\phi$ for $T_c$ = %5.4g $\omega_E$' % (tc/w_e),
                  fontsize=22)
        plt.savefig('./phi_plots/phi_tc_lam_%g_D_%g_we_%g.pdf'
                    % (llam, D, w_e),
                    bbox_inches='tight')
        plt.show()

        print('*********************\ng: %g, w_e: %3.2g, D: %g' % (g, w_e, D))
        print('lambda = %g\n*********************' % llam)
        print('dom_lim = %i' % dom_lim)
        print('Converged tc/w_e: %3.2g' % (tc/w_e))

    return tc
