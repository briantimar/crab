#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 17 02:44:14 2018

@author: btimar
"""
import numpy as np


class Constraint(object):
    pass

class MaxConstraint(Constraint):
    """Max value of function is constrained during the evolution"""
    
    def __init__(self,maxval):
        self.maxval=maxval
        
    def penalty(self,y):
        return y>self.maxval
        
class CostConstr(object):
    pass

class Basis(object):
    
    def __init__(self):
        pass
    
    def get_init_params(self):
        if self._init_params is None:
            raise ValueError("Params have not been initialized!")
        return self._init_params

class TestBasis(Basis):
    """Used just to check that the minimization routine actually works.
        """
    def __init__(self, N):
        #dimensionality of the problem
        self.N=N
        self._init_params=None
    
    def set_init_params(self):
        self._init_params= np.random.rand(self.N)        


class RandomFourierBasis(Basis):
    """A basis of (real) fourier modes over a fixed time interval.
        At instantiation, the frequencies are selected randomly.
        
        The optimization is over the amplitudes only.
        Amplitudes are initialized at random.
        
        """
        
    _bc_types = ['none', 'zero', 'unit']
    _bc_enforcement_types=['sine']        


    def __init__(self, Nmode, T, rscaling=1.0,bc='none', bc_enforcement = 'sine'):
        """Nmode = number of frequencies to allow
            T = the fundamental period.  The time-domain function is defined on [0, T]
            rscaling = magnitude of the fluctuation allowed in the frequencies
            bc = what boundary conditions the time-domain function has to satisfy.
                Possible values:
                    'none' --> no constraint. DC value is included in parameters to optimize over.
                    'zero' --> the function has to vanish at t=0, T.
                         f = (bc enf) * (sum of sinusoids)
                    'unit' --> the function must equal 1 at t=0, T. 
                          f = (bc enf) * (sum of sinusoids) + 1
            bc_enforcement: how the boundary conditions are enforced. Allowed values:
                    'sine' """
                  
        if bc not in RandomFourierBasis._bc_types:
            raise NotImplementedError
        if bc_enforcement not in RandomFourierBasis._bc_enforcement_types:
            raise NotImplementedError
                    
        #number of modes
        self.Nmode = Nmode
        #number of params ( 1 DC value + 2 * (Nmode -1) nonzero-freq amplitudes)
        self.N = 2 * Nmode -1 
        #interval of evolution
        self.T = T
        #shape of parameter array
        self.shape = (self.N,)    
        #amplitude of frequency fluctuations
        self.rscaling=rscaling
        self.omega_base = 2 * np.pi / self.T
        self.harmonics = np.array([ self.omega_base * k for k in range(0, self.Nmode)])
        self._frequencies=None
        self.amplitudes = None       
        self._init_params = None
        self._set_randomized_frequencies()
        self.bc=bc
        self.bc_enforcement = bc_enforcement
        assert len(self.shape)==1
        
    def _set_randomized_frequencies(self):
        ### add a random value to all the nonzero frequencies
        r = (np.random.rand(self.Nmode-1) - 0.5) * self.rscaling
        self._frequencies = self.harmonics.copy()
        self._frequencies[1:] += self.omega_base * r
    
    def get_frequencies(self):
        ### return all the frequencies (including zero)
        return self._frequencies
    
    def set_init_params(self, scaling = 1):
        """Choose initial values for the fourier amplitudes.
        
        Uniform over the interval [-scaling, scaling].
        If the DC amplitude is allowed to vary, it is set to 1.0.
        For consistency with scipy minimize routine all param arrays are 1d.
        The first Nmode are the cosine amplitudes and the second Nmode-1 are the sine amplitudes
        The zeroth component is the DC"""
        self.amplitudes = (np.random.rand(self.N) - 0.5)*scaling
        self.amplitudes[0] = 1.0
        
        assert self.amplitudes.shape == self.shape
        self._init_params = self.amplitudes
    
    def _gen_sinusoid_matrix(self,t):
        """evaluate all the fourier basis elements at specified times.
        t = scalar or 1d numpy array.
        Returns: array of size (N, Nt), Nt being the number of times provided
        
        """
        try:
            Nt = len(t)
            if t.ndim !=1:
                raise ValueError("time array should be one-dimensional")
        except TypeError:
            Nt = 1
        M = np.empty((self.N, Nt))
        t = t.reshape((1,Nt)).repeat(self.Nmode,axis=0)
        omega = self.get_frequencies().reshape((Nmode, 1)).repeat(Nt, axis=1)
        phi = t * omega
        M[:Nmode, :] = np.cos(phi)
        M[Nmode:, :] = np.sin(phi[1:,:])
        return M
    
    def _apply_bc(self, f_eval, t):
        if self.bc=='none':
            return f_eval
        else:
            enforcer = self._get_enforcer()
            if self.bc=='zero':
                return self._enforce_zero_bc(f_eval,t, enforcer)
            elif self.bc=='unit':
                return self._enforce_unit_bc(f_eval, t, enforcer)
        raise NotImplementedError

    def _get_enforcer(self):
        if self.bc_enforcement == 'sine':
            return lambda t: np.sin(np.pi * t / (self.T))
        else:
            raise NotImplementedError
    
    def _enforce_zero_bc(self,f_eval, t, enforcer):
        return f_eval * enforcer(t)
    
    def eval_time_fn(self, params, t):
        """ Using the specified params as amplitudes, compute the time-values.
        t = a scalar or 1d numpy array"""
        if params.shape != self.shape:
            raise ValueError("Param array has wrong shape")
        M = self._gen_sinusoid_matrix(t)
        assert M.shape[0] == self.N
        Nt=M.shape[1]
        params = params.reshape((self.N,1)).repeat(Nt,axis=1)
        f_evaluated= np.sum(M*params,axis=0)
        assert (f_evaluated.ndim==1) and len(f_evaluated==Nt) 
        
        #apply boundary conditions, if any
        f_evaluated = self._apply_bc(f_evaluated, t)
        return f_evaluated
    
    def get_time_function(self, params):
        """ Given a particular set of params, return the associated function of time.
        t is allowed to be an array of times."""
        def time_fn(t):
            return self.eval_time_fn(params, t)
        return time_fn
    
    
    
class TestCostConstr(CostConstr):
    """Returns a very simple cost function -- just some polynomial of the 'basis' params"""
    
    def __init__(self):
        pass
    
    def get_cost_function(self,h,basis):
        return lambda x : np.sum(x**2) * (np.sum(x**2)-1)

class KineticCostConstr(CostConstr):
    """ The cost of a function is its kinetic energy. Normalization is required.
            This class is meant as a test. 
            """
    
    def __init__(self, T,nstep=int(1E4), LM_norm=1.0):
        self.T = T
        self.nstep=nstep
        self.times = np.linspace(0, T, nstep)
        self.dt = np.diff(self.times)[0]
        
        #lagrange multiplier for the norm cost
        self.LM_norm = 1.0
        
    def integrate(self, farray):
        return np.sum( farray ) * self.dt /T
        
    def _kinetic_cost(self,f):
        """f = some function of time"""
        ft = f(self.times)
        ddf = np.diff(ft,n=2)
        return self.integrate(- ft[1:-1] * ddf )
        
    def _norm_cost(self,f):
        ft = f(self.times)
        return np.abs(self.integrate( np.abs(ft)**2) -1)
    
    def cost(self,f):
        """ Defines the total cost assigned to a particular function of time, f."""
        return self._kinetic_cost(f) + self.LM_norm * self._norm_cost(f)
    
    def get_cost_function(self, h, basis):
        """ h = initial guess for the function.
            The function that's returned is passed to the scipy optimize routine, which uses 1d arrays for all parameters"""
        
        def _cost(params):
            if params.ndim != 1:
                raise ValueError("expecting 1d param array")
            g = basis.get_time_function(params)
            f = lambda t : h(t) * g(t)
            return self.cost(f)
        return _cost



class Trial(object):
    """A particular instance of optimization with a fixed basis"""
    
    methods = ['Nelder-Mead']
    
    def __init__(self, basis, C, h, **opt_args):
        """C = the cost function
            h = zeroth-order pulse sequence
            opt_args = args for the scipy optimizer"""
        self.basis = basis
        self.C=C
        self.h=h
        self.opt_args = opt_args
      
    def do_minimize(self, method='Nelder-Mead'):
        if method not in Trial.methods:
            raise NotImplementedError
        from optimize import do_nm_minimize
        #starting params
        x0 = self.basis.get_init_params()
        #the function which nm attempts to minimize
        f = self.C.get_cost_function(self.h, self.basis)
        return do_nm_minimize(f, x0, **self.opt_args)



class CRABOptimizer(object):
    """Holds the parameters and trials of a particular crab optimization problem"""
    def __init__(self, C, h, basis ):
        """C: returns the overall cost function (including constraints) which is supposed to be minimized. Accepts as input a set of pulse sequences.
             h: zeroth-order guess for the pulse sequences
             basis: object containing list of basis functions which are specified by some finite number of params
             """
        self.C=C
        self.h=h
        self.basis=basis
           
    def do_trial(self):
        self.basis.set_initial_params()
        self.fixed_basis_optimize()
        
    def fixed_basis_optimize(self):
        trial = Trial(self.basis, self.C, self.h)
        res = trial.do_minimize()
        
#        
#Nmode=10
#T=1.0
#basis = RandomFourierBasis(Nmode, 2*T,rscaling=0.0, bc='zero')
#cConstr= KineticCostConstr(T, nstep=10000,LM_norm=1.0)
##initial wf guess 
#h = lambda t: np.ones(len(t))
#basis.set_init_params()
#trial = Trial(basis,cConstr,h)
#res=trial.do_minimize()
#params_fit = res.x
#times = np.linspace(0, 2*T, 100)
#g_final = basis.get_time_function(params_fit)
#plt.plot(times, h(times)*g_final(times))
#plt.plot(times, h(times))
#f = lambda t: h(t) * g_final(t)
#print(cConstr._kinetic_cost(f))
#print(cConstr._norm_cost(f))
#        
#    