#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Basis types for CRAB. Only a fourier basis currently.
"""
import numpy as np

class Basis(object):
    
    def __init__(self):
        pass
    
    def get_init_params(self):
        if self._init_params is None:
            raise ValueError("Params have not been initialized!")
        return self._init_params

    
    def get_signal(self, h,params):
        """ Returns the actual time-signal f(t) which is fed to the cost function"""
        if params.ndim != 1:
                raise ValueError("expecting 1d param array")
        g = self.get_time_function(params)
        f = lambda t : h(t) * g(t)
        return f


class Signal(object):
    """ Stores one or many pulse sequences."""

    def __init__(self, **sig_dict):
        self._sig_dict = sig_dict
        
    

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
    
    def set_frequencies(self,f):
        """Set the frequencies by hand -- only allowed if a random set hasn't already been generated"""
        if self._frequencies is not None:
            print("Can't set frequencies after random initialization")
            return
        if len(f) != self.Nmode:
            raise ValueError("Number of frequencies should match Nmode")
        self._frequencies = f
        
    def copy(self):
        b= RandomFourierBasis(self.Nmode, self.T, rscaling=self.rscaling,bc=self.bc, bc_enforcement =self.bc_enforcement)
        if self.get_frequencies() is not None:
            b.set_frequencies(self.get_frequencies())
        return b
    
    
    def _set_init_params(self, scaling = 1):
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
    
    def initialize(self):
        """Set new random frequencies and amplitudes."""
        self._set_init_params()
        self._set_randomized_frequencies()
    
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
            t=np.array([t])
        
        t = t.reshape((1,Nt)).repeat(self.Nmode,axis=0)
        M = np.empty((self.N, Nt))
        
        omega = self.get_frequencies().reshape((self.Nmode, 1)).repeat(Nt, axis=1)
        phi = t * omega
        M[:self.Nmode, :] = np.cos(phi)
        M[self.Nmode:, :] = np.sin(phi[1:,:])
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
    
    def _enforce_unit_bc(self, f_eval, t, enforcer):
        return 1.0 + self._enforce_zero_bc( f_eval, t, enforcer)
    
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
        assert (f_evaluated.ndim==1) and len(f_evaluated)==Nt 
        #apply boundary conditions, if any
        f_evaluated = self._apply_bc(f_evaluated, t)
        #return a scalar if provided a scalar
        if Nt==1:
            return f_evaluated[0]
        return f_evaluated
    
    def get_time_function(self, params):
        """ Given a particular set of params, return the associated function of time.
        t is allowed to be an array of times."""
        def time_fn(t):
            return self.eval_time_fn(params, t)
        return time_fn
    