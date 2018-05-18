#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 17 02:44:14 2018

@author: btimar
"""
import numpy as np
from quspin.operators import hamiltonian
from scipy.integrate import quad

def integrate(f, a, b):
    return quad(f, a, b)[0]

def format_state(s):
    """Checks for array type and tries to convert to 1d array.
       Assumes a pure state, i.e. vector, s.
       
       Returns: one-dimensional numpy array."""       
    if not isinstance(s, np.ndarray):
        raise TypeError("States must be implemented as numpy arrays")
    if s.ndim == 1:
        return s
    if s.ndim != 2:
        raise ValueError("State array has too many dimensions")
    l,dummy = max(s.shape), min(s.shape)
    if dummy != 1:
        raise ValueError("Conversion into 1d array is ambiguous. For a pure state, one of the array axes ought to be of length 1.")
    return s.reshape(l)
        
def overlap(s1,s2):
    """Returns the overlap <s1|s2>.
    s1, s2 = two pure states.
    """
    s1 = format_state(s1)
    s2 = format_state(s2)
    if len(s1)!=len(s2):
        raise ValueError("Input states must be the same shape.")
    return np.sum(s1.conjugate() * s2) 

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
    

class CostConstr(object):
    
    def get_cost_function(self, h, basis):
        """ h = initial guess for the function.
            The function that's returned is passed to the scipy optimize routine, which uses 1d arrays for all parameters"""
        
        def _cost(params):
            return self.cost(basis.get_signal(h, params))
        return _cost


    
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
        return np.sum( farray ) * self.dt /self.T
        
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
    

    
    
class QEvolver(object):
    """ Wrapper for quspin time-evolution.
        """
    _methods = ['direct']
    
    def __init__(self, basis, static=None, psi0=None, interval=None):
        """ basis: the quspin basis object used to construct the hamiltonian
           static: static opstr list for the hamiltonian
           psi: initial state
           interval: time interval of evolution"""
        
        ###basis
        self._basis = basis        
        ##dimensionality of the hilbert space
        self.D=basis.Ns
        ### number of sites
        self.N = basis.N
        ##Holds the time-evolved states
        self._psit=None
        ###static opstr list for the hamiltonian
        self._static= []
        ###dynamic opstr list for the hamiltonian
        self._dynamic = []
        ### hamiltonian
        self._H = None
        self._has_altered_hamiltonian=False
        
        self._direct_solver_args = dict(Nt=10, dN=10, nmax=5)
        if static is not None:
            self.set_static(static)
        if psi0 is not None:
            self.set_psi0(psi0)
        if interval is not None:
            self.set_interval(interval)
        
    def set_psi0(self, psi0):
        """Assigns an initial pure state psi0.
           psi0 = np array"""
        if psi0.ndim !=1:
            raise ValueError("Not a valid pure state")
        if len(psi0) != self.D:
            raise ValueError("psi has incorrect dimensions")
        
        self._psi0 = psi0
        
    def set_static(self,static):
        """Updates the static list"""
        self._static=static
        self._has_altered_hamiltonian=True
        
    def set_dynamic(self, dynamic):
        """dynamic = a dynamic opstring list in quspin format."""
        self._dynamic = dynamic
        self._has_altered_hamiltonian=True
             
    def _set_psit(self, psit):
        
        if psit.shape[0] != self.D:
            raise ValueError("Incorrect dimension of returned state")
        self._psit = psit
        
    def get_psit(self, method='direct', **solver_args):
        """Return the time-evolved state, under the current hamiltonian params"""
        self.evolve(method=method, **solver_args)
        return self._psit
        
    def set_hamiltonian(self):
        """Assigns hamiltonian to be used for each evolution. Need not be hermitian.
        """
        self._H=hamiltonian(self._static, self._dynamic, basis=self._basis)
    
    def get_hamiltonian(self):
        if self._has_altered_hamiltonian:
            self.set_hamiltonian()
            self._has_altered_hamiltonian=False
        return self._H
    
    def set_interval(self, interval):
        """ Interval over which evolution will take place."""
        self._interval = interval
        self.ti=interval[0]
        self.tf=interval[1]
    
    def set_direct_solver_args(self,Nt,dN,nmax):
        """Nt   Number of time-slices to compute the state at. If the integrator throws RunTimeError this will
            be increased iteratively by dN until integrator succeeds or nmax is reached."""
        self._direct_solver_args = dict(Nt=Nt,dN=dN,nmax=nmax)
    
    def _evolve_direct(self,Nt=10,dN=10,nmax=5,**solver_args):
        for __ in range(nmax):
            times = np.linspace(self.ti,self.tf,Nt)
            try:
                H=self.get_hamiltonian()
                psit= H.evolve(self._psi0, self.ti,times,**solver_args)
                self._set_psit(psit)
                return
            except RuntimeError:
                Nt += dN
                print("Integration failed, trying Nt = {0}.format(Nt)")
        raise RuntimeError("Integration failed")
            
    def evolve(self, method='direct', **solver_args):
        """Time-evolve the state, from beginning to end of the interval, and updates stored psit.
            If method='direct':
                Calls h.evolve(). No parallelization tricks are applied.
                """
        if len(self._dynamic)==0:
            raise ValueError("Dynamic list has not been added!")
        if method not in QEvolver._methods:
            raise NotImplementedError
        if method=='direct':
            return self._evolve_direct(**self._direct_solver_args,**solver_args)
        raise NotImplementedError
        

class QEvCostConstr(CostConstr):
    """Defines a cost function based on some quantum evolution."""
    
    def __init__(self, basis, static, psi0,  interval, costfn):
        """
        psi0 = initial state, np array
        H = quspin hamiltonian
        interval = interval of time evolution.
        costfn:  some function that takes a D x Nt array of evolved states as input (D being the dimensionality) and returns a single real number which is to be minimized.
        """
        self.basis=basis
        self._qevolver = QEvolver(basis,static,psi0,interval)
        self._primary_costfn = costfn
        self._interval = interval
        self.ti = interval[0]
        self.tf = interval[1]
                
        ## (LM, const_cost) pairs, where LM is a lagrange multiplier
        ### (actually just a fixed weight) and const_cost is the cost function which defines the constraint.
        ###  the constraint functions are allowed as inputs: f, the drive signal, and psit
        self._constraints = []
    
    def make_dynamic(self,f):
        raise NotImplementedError("Use a subclass")
    
    def _get_psit(self, f):
        """Compute the time-evolved state."""
        dynamic = self.make_dynamic(f)
        self._qevolver.set_dynamic(dynamic)
        return self._qevolver.get_psit()
    
    def cost(self,f):
        """Returns the cost of a drive signal f(t)"""
        #compute the time-evolved state(s)
        psit=self._get_psit(f)
        c=self._primary_costfn(psit)
        for LM,cf in self._constraints:
            c += LM * cf(f,psit)
        return c
    
    def _integral_cost(self,f, psit):
        """Cost is defined as the time-integral of |f|^2"""
        fsq = lambda f: np.abs(f)**2
        return integrate(fsq, self.ti, self.tf)
    
    def add_integral_constraint(self, lm):
        self._constraints.append((lm, self._integral_cost))

    def _fidelity_cost(psit, psi_target):
        """Defines the cost as the infidelity of the final state with respect to some target."""
        return 1 - np.abs(overlap(psit[:,-1], psi_target))**2
        
    
class QEvFidelityCC(QEvCostConstr):
    """ Returns a cost function based on infidelity with a particular target state"""
    
    def __init__(self, basis, static, psi0, interval, psi_target):
        primary_cost_fn = lambda psit: QEvCostConstr._fidelity_cost(psit, psi_target)
        self.psi_target = psi_target
        QEvCostConstr.__init__(self, basis, static, psi0, interval, primary_cost_fn)


class UniformFieldFidelityCC(QEvFidelityCC):
    """Here the input pulse sequence f(t) couples to a term
          sum_i f(t) O_i
          in the hamiltonian, where O_i are single-site operators."""

    def __init__(self, basis, static, psi0, interval, psi_target, coupling_opstr):
        """ coupling_opstr = the label of the local operator to which the drive field couples.
                Examples: 'z', 'n', 'x'.
                """
        QEvFidelityCC.__init__(self, basis, static,psi0,interval,psi_target)
        self._coupling_opstr = coupling_opstr
    
    def make_dynamic(self, f):
        """ Returns the dynamic opstr list to pass to quspin constructor."""
        #number of sites
        N=self.basis.N
        coupling = [[1.0, i] for i in range(N)]
        drive_args = []
        return [ [self._coupling_opstr, coupling, f, drive_args]]
        

class QubitZSweep(UniformFieldFidelityCC):
    """Test case. A single qubit with constant X field, variable Z field."""
    
    
    def __init__(self, Omega, psi0, interval, psi_target):
        static = [['x', [[Omega, 0]]]]
        from quspin.basis import spin_basis_1d
        basis = spin_basis_1d(1)
        UniformFieldFidelityCC.__init__(self, basis, static,psi0,interval,psi_target, 'z')
    
    
######################
        
        
    
    
    
    
######################



class Trial(object):
    """A particular instance of optimization with a fixed basis"""
    
    _methods = ['Nelder-Mead']
    
    def __init__(self, basis, C, h, **opt_args):
        """C = the cost function
            h = zeroth-order pulse sequence
            opt_args = args for the scipy optimizer"""
        self.basis = basis
        self.C=C
        self.h=h
        self.opt_args = opt_args
      
    def do_minimize(self, method='Nelder-Mead'):
        if method not in Trial._methods:
            raise NotImplementedError
        from optimize import do_nm_minimize
        #starting params
        x0 = self.basis.get_init_params()
        #the function which nm attempts to minimize
        f = self.C.get_cost_function(self.h, self.basis)
        return do_nm_minimize(f, x0, **self.opt_args)


class CRABOptimizer(object):
    """Holds the parameters and trials of a particular crab optimization problem"""
    def __init__(self, cConstr, h, basis, **opt_args ):
        """Cconstr: returns the overall cost function (including constraints) which is supposed to be minimized. Accepts as input a set of pulse sequences.
             h: zeroth-order guess for the pulse sequences
             basis: object containing list of basis functions which are specified by some finite number of params
             """
        self.cConstr = cConstr
        self.h=h
        self.basis=basis
        self.trial = Trial(basis,cConstr,h,**opt_args)
        self.opt_args = opt_args   
        
    def do_trial(self):
        """ Obtain a fresh basis and run minimization routine."""
        self.basis.initialize()
        return self.trial.do_minimize()
        
    def pocket_CRAB(self, Ntrial=20):
        """Run Ntrial different optimization routines, picking a new basis each time.
        Return the basis and params which yielded the lowest cost.
        returns tuple:
            (best basis, best params, best cost)
        
        """
        best_basis = None
        best_cost =None
        best_params = None
        for __ in range(Ntrial):
            res=self.do_trial()
            ##final cost value
            if best_cost is None or (res.fun < best_cost):
                best_cost = res.fun
                best_params = res.x
                best_basis = self.basis.copy()
        return (best_basis, best_params, best_cost)
        
        
        
        
        
        
        
        