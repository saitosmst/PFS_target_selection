#! /usr/bin/env python3
# -*- coding:utf-8 -*-

r"""
    author: SHUN SAITO (Missouri S&T)

    Time-stamp: <Fri Sep 18 09:32:48 CDT 2020>

    This code is developed for the target-selection purpose 
    in the PFS cosmology working group.


    The code is highly relying on the EL-COSMOS catalog 
    (Saito et al. 2020) which predicts the [OII] fluxes
    as well as broadband photometries from HSC, (g, r, i, y, z).

    Version history:
        0.0    initial migration from jupyter notebooks
"""

__author__ = 'Shun Saito'
__version__ = '0.0'


import numpy as np
from astropy import units as u
from astropy.cosmology import FlatLambdaCDM
from astropy.constants import c
from astropy.io import fits
from scipy import integrate, interpolate
from scipy.stats import poisson

# hard-coded constants and setting
## PFS
AREA_PFSFoV = 1.098 # Area of the Field of View in one visit [deg^2]
AREA_PFS = 1464     # Area of the total PFS cosmology footprint [deg^2]
NUM_FIBER = 2394    # Number of fibers available in one visit
## EL-COSMOS
AREA_CMC = 1.38     # Area of the COSMOS-2015 catalog [deg^2]
## Fiducial cosmology from astropy
cp = FlatLambdaCDM(H0=67.77, Om0=0.307115)
littleh = cp.H0.value/100
Mpcph = u.def_unit('Mpcph', u.Mpc/littleh) #[Mpc/h]
dVc_dzdOmega = lambda x: cp.differential_comoving_volume(x).to(Mpcph**3/u.sr).value #[(Mpc/h)^3/sr]


class FisherLDA(object):
    r"""
    A simple class to compute the Fisher Linear Discriminant Analysis.

    Input
    ----------------

    Attributes
    ----------------

    Methods
    ----------------

    """
    def __init__(self, ndim, y_A, y_B):
        self.ndim = ndim

        try:
            assert len(y_A.shape) == 2
            assert y_A.shape[1] == ndim
            assert len(y_B.shape) == 2
            assert y_B.shape[1] == ndim
        except AssertionError:
            msg = 'The input arrays have wrong shape.'
            raise ValueError(msg)

        num_A = y_A.shape[0]
        num_B = y_B.shape[0]
        y = np.zeros(shape=(num_A+num_B,ndim))
        y[:num_A,:] = y_A
        y[num_A:,:] = y_B
        y_mean = np.array([np.mean(y[:,i]) for i in range(ndim)])
        self.y = y
        
        self.yA_mean = np.array([np.mean(y_A[:,i]) for i in range(ndim)])
        self.yB_mean = np.array([np.mean(y_B[:,i]) for i in range(ndim)])
        
        self.covmat = np.zeros(shape=(ndim,ndim))
        for i in range(ndim):
            for j in range(ndim):
                self.covmat[i,j] = np.sum( (y[:,i]-y_mean[i])*(y[:,j]-y_mean[j]) )
        self.covmat[:,:] /= (num_A+num_B-1)
        self.invcov = np.linalg.inv(self.covmat)

        # compute the Fisher LDA (see Eq.(1) in Raichoor+2016)
        self.x_fi = np.zeros(shape=(num_A+num_B))
        t = np.sqrt(num_A*num_B)/(num_A+num_B)*np.dot(np.transpose(self.yA_mean-self.yB_mean), inv_cov)
        self.x_fi = np.array([ t[i]*y[:,i] for i in range(ndim) ])

        return
    
