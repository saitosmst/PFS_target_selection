#! /usr/bin/env python3
# -*- coding:utf-8 -*-

r"""
    author: SHUN SAITO (Missouri S&T)

    Time-stamp: <Wed Sep 23 09:48:51 CDT 2020>

    This code is developed for the target-selection purpose 
    in the PFS cosmology working group.

    This code provides a simple, general class to compute 
    the Fisher LDA. 

    Version history:
        0.0    initial migration from jupyter notebooks
"""

__author__ = 'Shun Saito'
__version__ = '0.0'


import numpy as np

class FisherLDA(object):
    r"""
    [Sorry, under construction!]

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
    
