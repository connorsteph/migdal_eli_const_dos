# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 11:40:00 2017

@author: Connor
"""

import numpy as np
from tc_calc import tc_calc
import matplotlib.pyplot as plt
import tc_func as tf
from zeta_solver import zeta_solver
from phi_solver import init_summand
from tc_calc import init_phi

"""
Problem params
"""
D = 16
w_e = 16/2.5
lam_want = 2
g = np.sqrt(lam_want*D*w_e/2)

"""
Algorithm params
"""

dom_lim = 200
p_damp = 0.3
maxiter = 50
tol = 1e-5
damp = 0.3
s = 1

t = 0.1*w_e
#domain = np.linspace(0.5, 20, 10)
#plt.figure()
#plt.grid(True)
#plt.plot(domain, [tc_calc(np.sqrt(lam_want*(Q*w_e)*w_e/2), w_e, Q*w_e,
#                          dom_lim, maxiter=maxiter,
#                          tol=tol, p_tol=5e-5, t_tol=5e-2, plot=False)/w_e
#                  for Q in domain])
#plt.title(r'$T_c$ as a function of $\frac{D}{\omega_E}$', fontsize=22)
#plt.ylabel(r'$\frac{T_c}{\omega_E}$', fontsize=18)
#plt.xlabel(r'$\frac{D}{\omega_E}$', fontsize=18)
#plt.savefig('tc_plot_lam_%g.pdf' % lam_want, bbox_inches='tight')


tc_calc(g, w_e, D,
        dom_lim, maxiter=maxiter,
        tol=tol, p_tol=5e-5, t_tol=5e-2, plot=False, iprint=False)

#num = 100
#llam = 2*g**2/(D*w_e)
#zeta = zeta_solver(t, g, w_e, num, D, damp=0.3, maxiter=150, iprint=False)
#plt.figure()
#plt.grid(True)
#domain = tf.m_array(5, num)
#for w_e in [1, 10, 20,100]:
#    m=30
#    y = [init_summand(w_n, tf.freq_m(m, t), init_phi, zeta, w_e, t, D)
#         for w_n in tf.freq_array(5, num, t)]
#    plt.plot([tf.freq_m(m, t) for m in domain], y, label=('w_e = %g' % w_e))
#plt.legend(loc='best')
#plt.xlabel('w_n')
###
#plt.figure()
#plt.grid(True)
#domain = np.arange(1, 200, 10)
#for w_e in [1, 10, 20, 100]:
#    m = 30
#    y = [llam*np.pi*t*(tf.matsu_sum(1, M, t, init_summand, tf.freq_m(m, t),
#                       init_phi, zeta, w_e, t, D))
#         for M in domain]
#    plt.plot(domain, y, label=('w_e = %g' % w_e))
#plt.legend(loc='best')


#plt.figure()
#domain = np.linspace(1, 21)
#for q in [100*s, 200*s]:
#    y=[]
#    for t in domain:
#        zeta = zeta_solver(t, np.sqrt(q), w_e, 50, D,
#                                     damp=0.3, maxiter=150, iprint=False)
#        y.append(zeta(tf.freq_m(1, t))/tf.freq_m(1, t))
#    plt.plot(domain, y, label=('q = %g' % (q)))
#plt.plot(domain, [(2*(200*s)/(D*w_e))+1 for t in domain], label='correct')
##plt.ylim([0, 1.5])
#plt.xlabel('t/w_e')
#plt.legend(loc='best')
#plt.grid(True)
#zeta = zeta_solver(t, g, w_e, 150, D, damp=0.3, maxiter=150, iprint=True)
