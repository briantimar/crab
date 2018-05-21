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
    
    def get_full_time_function(self, h,params):
        """ Returns the actual time-signal f(t) which is fed to the cost function.
             Currently, this is h(t) * g(t), where g is returned by get_time_function
             
             h(t): the zeroth-order time function."""
        if params.ndim != 1:
                raise ValueError("expecting 1d param array")
        g = self.get_time_function(params)
        f = lambda t : h(t) * g(t)
        return f

    def get_signal(self, sig0, params):
        """ Returns a Signal() object wrapping the requested time function.
        sig0 = Signal containg zeroth-order pulse seqeunce.
        """
        
        if len(sig0.keys) !=1:
            raise ValueError("Individual Basis() can only accept one time-profile.")
        try:
            if sig0.keys[0] != self.label:
                print(sig0.keys[0], self.label)
                raise ValueError("input Signal does not match label of this basis.")
        except AttributeError:
            raise ValueError("Can't create Signal without labeling this basis")
        h = sig0.get_individual(self.label)
        f=self.get_full_time_function(h,params)
        sd=dict()
        if self.label is None:
            raise ValueError("Can't create Signal without labeling this basis")
        sd[self.label] = f
    
                
        return Signal(**sd)


class Signal(object):
    """ Stores one or many pulse sequences."""

    def __init__(self, **sig_dict):
        """The dictionary should map signal label strings to time-functions."""
        self._sig_dict = sig_dict
        self._keys = list(sig_dict.keys())
        
    def add_individual(self,k, sig):
        """Add a label k and corresponding time-function"""
        self._sig_dict[k] = sig
        self._keys.append(k)
        
    def get_individual(self,k):
        """ Return an individual time-function"""
        return self._sig_dict[k]
    
    @property
    def keys(self):
        return self._keys
    
    def join(self, signal):
        """Pull in signals from another signal object."""
        for k in signal.keys:
            self.add_individual(k, signal.get_individual(k))
            self._keys.append(k)
    

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


    def __init__(self, Nmode, ti, tf, rscaling=1.0,bc='none', bc_enforcement = 'sine', label=None):
        """Nmode = number of frequencies to allow
        ti, tf: initial/final endpoints defining the interval of evolution
            T = the fundamental period = tf - ti
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
        self.ti=ti
        self.tf=tf
        assert tf > ti
        #interval of evolution
        self.T = tf - ti
        #shape of parameter array
        self.shape = (self.N,)    
        #amplitude of frequency fluctuations
        self.rscaling=rscaling
        self.omega_base = 2 * np.pi / self.T
        self.harmonics = np.array([ self.omega_base * k for k in range(0, self.Nmode)])
        self._frequencies=None
        self.amplitudes = None       
        self._init_params = None
        #labels the operator this basis will couple to
        self.label=label
        self.bc=bc
        self.bc_enforcement = bc_enforcement
        assert len(self.shape)==1
        
        
    def set_label(self, k):
        """Assign opstr label. This should match one of the labels of the cConstr it's passed to."""
        self.label =k
        
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
        """Return a copy of this basis, with the same label and frequencies."""
        b= RandomFourierBasis(self.Nmode, self.ti, self.tf, rscaling=self.rscaling,bc=self.bc, bc_enforcement =self.bc_enforcement, label=self.label)
        if self.get_frequencies() is not None:
            b.set_frequencies(self.get_frequencies())
        b.set_label(self.label)
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
            return lambda t: np.sin(np.pi * (t-self.ti) / (self.T))
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
    
    
class MultipleSignalBasis(object):
    """A container for multiple pulse sequences, each of which uses the same basis type."""
    
    def __init__(self, **basis_dict):
        """ basis_dict: a dictionary mapping descriptive keys (can be anything) to basis objects."""
        self._basis_dict = basis_dict
        self._keys = list(basis_dict.keys())
        self._basis_list = list(basis_dict.values())
        self._nsig = len(basis_dict)
        #total number of parameters
        self._nparam_list = [ basis.N for basis in self._basis_list]
        self._nparam = sum(self._nparam_list)
        
        
    def _split_params(self, params):
        """ Given a 1d param array, returns a dictionary mapping keys in self._keys to appropriate sub-arrays of the provided param array.
             Default behavior is to just pass them to them to the various bases in the order determined by self._keys."""
        sp=dict()
        q=0
        if len(params) != self._nparam:
            raise ValueError("incorrect number of parameters")
        for i in range(len(self._keys)):
            sp[self._keys[i]] = params[q:q+self._nparam_list[i]]
            q+=self._nparam_list[i]
        return sp
    
    def get_basis(self, k):
        return self._basis_dict[k]
    
    def _gen_signals(self, sig_trial, param_split_dict):
        """ Given a Signal() sig_trial, and param dictionary, the corresponding time functions are computed from each Basis, and returned in a new Signal()."""
        sigs = Signal()
        for k in self._keys:
            params = param_split_dict[k]
            h = sig_trial.get_individual(k)
            sigs.add_individual(k, self.get_basis(k).get_full_time_function(h, params))
        return sigs
    
    def get_signal(self,sig_trial, params):
        """ Return, for a given set of params, corresponding signals from all bases.
            params: a single, one-dimensional numpy array which contains params for *all* bases.
            sig_trial: a Signal() object containing the trial pulse shapes"""
        if set(sig_trial.keys) != set(self._keys):
            raise ValueError("Signal labels do not match this MultipleSignalBasis.")
        params_split = self._split_params(params)
        return self._gen_signals(sig_trial, params_split)
        
    def initialize(self):
        """Initialize each Basis"""
        for k in self._keys:
            self.get_basis(k).initialize()
            
    def copy(self):
        """returns new MSB by copying each individual basis."""
        bdict = dict()
        for k in self._keys:
            bdict[k] = self._basis_dict[k].copy()
        return MultipleSignalBasis(**bdict)        
    
    def get_init_params(self):
        """Return the full 1d vector of params which is passed to scipy.optimize."""
        params = []
        for k in self._keys:
            params += list(self.get_basis(k).get_init_params())
        return np.array(params)
        
        
        
        
        
    
    
    
    