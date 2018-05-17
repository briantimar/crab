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
        

class Problem(object):
    """Defined by a particular cost function, f, and some constraints that it has to satisfy."""
    pass
        


class Trial(object):
    
    def __init__(self, basis, C, h):
        self.basis = basis
        self.C=C
        self.h=h
    def 


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
        self.basis.get_initial_params()
        self.basis.get_basis()
        self.fixed_basis_optimize()
        
    def fixed_basis_optimize(self):
        trial = Trial(self.basis, self.C, self.h)
        
        
        
        
        
        
        
        
        
        
        
        
    