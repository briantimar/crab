#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 18 22:05:17 2018

@author: btimar
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('/Users/btimar/Documents/ryd-theory-code/python_code/')
sys.path.append('/Users/btimar/Documents/ryd-theory-code/python_code/experiment_related')

from basis import MultipleSignalBasis, RandomFourierBasis, Signal
from crab import CRABOptimizer
from tools import get_hcb_basis

from experimental_params import Omega_sr, C6_sr
from ryd_base import get_r6_1d_static_nn
a=5.0
Vnn=C6_sr/(a**6)
Omega=Omega_sr

L=4
ktrunc=L
bc='periodic'
dtype=np.float64
static = get_r6_1d_static_nn(Vnn, L, ktrunc, bc)

from sweep_tools import quasi_gaussian_window, soft_tanh_step

#sweep time in us
T=5.0
ti=-T/2
tf=T/2
##initial guess for the drive sequnces
#sweep values of omega
x_guess = quasi_gaussian_window(0.0, Omega/2.0, T/4, 6)
#sweep values of delta
n_guess = soft_tanh_step(.2*Vnn, -1.0*Vnn, .5*Vnn, T/6, 3)
t = np.linspace(ti,tf,100)
#plt.plot(t, x_guess(t), t, z_guess(t))

opstrs=['x','n']
s0 = Signal(x=x_guess, n=n_guess)

##define quantum basis and the initial state.
qbasis_full =get_hcb_basis(L)
qbasis_symm =get_hcb_basis(L,pblock=1,kblock=0)
from tools import get_Z2, get_uniform_state
psi0_full = (1.0/np.sqrt(2))*(get_Z2(qbasis_full, dtype, which=0)+get_Z2(qbasis_full, dtype, which=1))
proj = np.transpose(qbasis_symm.get_proj(dtype))
psi_target = proj.dot(psi0_full)
psi0 = proj.dot(get_uniform_state(qbasis_full,dtype,which=0))

from tools import norm
from numpy.testing import assert_almost_equal
assert_almost_equal(norm(psi0),1.0,decimal=10)
assert_almost_equal(norm(psi0),1.0,decimal=10)

##construct the cost function
from crab import UniformFieldFidelityCC
cConstr=UniformFieldFidelityCC(qbasis_symm, static,psi0,[ti,tf],psi_target, opstrs)

##set up fourier basis.
Nmode=3
bx=RandomFourierBasis(Nmode, T,bc='unit',label='x')
bn=RandomFourierBasis(Nmode, T,bc='unit',label='n')
msb=MultipleSignalBasis(x=bx,n=bn)

options_nm=dict(maxfev=100)

opt_args = dict(options=options_nm)
opt=CRABOptimizer(cConstr, s0, msb,**opt_args)

best_msb,params,cost=opt.pocket_CRAB(Ntrial=10)




