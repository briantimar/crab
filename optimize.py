#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 17 02:28:47 2018

@author: btimar

"""

import numpy as np

from scipy.optimize import minimize

def do_nm_minimize(f, x0,**args):
    """minimizer f via nelder-mead """

    return minimize(f, x0, method='Nelder-Mead',**args)
