#! /usr/bin/env python3
# -*- coding:utf-8 -*-

r"""
    author: SHUN SAITO (Missouri S&T)

    Time-stamp: <Wed Sep 23 09:42:25 CDT 2020>

    This code is developed for the target-selection purpose 
    in the PFS cosmology working group.

    The code is highly relying on the EL-COSMOS catalog 
    (Saito et al. 2020) which predicts the [OII] fluxes
    as well as broadband photometries from HSC, (g, r, i, y, z).

    The code computes the photometry errors assuming HSC limiting 
    magnitudes. The calculations for DECaLS and MzLS are simply 
    scaled from HSC assuming a rough scaling (see comment *1).

    Version history:
        0.0    initial migration from jupyter notebooks

    To-Do:
        - make the code more abstract
        - figure out the meaning of "factor" (see comment *2)
"""

__author__ = 'Shun Saito'
__version__ = '0.0'

import numpy as np
#astropy
from astropy import units as u
from astropy.cosmology import FlatLambdaCDM
from astropy.constants import c
from astropy.io import fits
from scipy import integrate, interpolate
from scipy.stats import poisson

def add_photometryerror(arr_galcat):
    limmagSN5_g_HSC = 25.2
    limmagSN5_r_HSC = 24.9
    limmagSN5_i_HSC = 24.5
    limmagSN5_y_HSC = 24.2
    limmagSN5_z_HSC = 24.2
    limmagSN5_g_DEcaLS = 24.2
    limmagSN5_r_DEcaLS = 23.3
    limmagSN5_z_DEcaLS = 22.5
    limmagSN5_g_MzLS = 23.5
    limmagSN5_r_MzLS = 23.0
    limmagSN5_z_MzLS = 22.4

    dtype_mags = np.dtype([('id', int), ('z_photo', float), ('R_e', float), 
                           ('gmag_org', float), ('rmag_org', float), ('imag_org', float), ('zmag_org', float), ('ymag_org', float), 
                           ('flux_OII', float), ('SNR_OII_PFS', float), ('SNR_OII_DESI', float), 
                           ('dgmag_HSC', float), ('drmag_HSC', float), ('dimag_HSC', float), ('dzmag_HSC', float), ('dymag_HSC', float), 
                           ('gflux_SN_HSC', float), ('rflux_SN_HSC', float), ('zflux_SN_HSC', float), ('iflux_SN_HSC', float), ('yflux_SN_HSC', float),
                           ('dgmag_DEcaLS', float), ('drmag_DEcaLS', float), ('dzmag_DEcaLS', float),
                           ('gflux_SN_DEcaLS', float), ('rflux_SN_DEcaLS', float), ('zflux_SN_DEcaLS', float), 
                           ('dgmag_MzLS', float), ('drmag_MzLS', float), ('dzmag_MzLS', float),
                           ('gflux_SN_MzLS', float), ('rflux_SN_MzLS', float), ('zflux_SN_MzLS', float)])
    arr_mags = np.zeros(shape=arr_galcat.shape[0], dtype=dtype_mags)
    
    arr_mags['id'] = arr_galcat['id']
    arr_mags['z_photo'] = arr_galcat['z_photo']
    arr_mags['R_e'] = arr_galcat['R_e']
    arr_mags['gmag_org'] = arr_galcat['g_hsc']
    arr_mags['rmag_org'] = arr_galcat['r_hsc']
    arr_mags['imag_org'] = arr_galcat['i_hsc']
    arr_mags['ymag_org'] = arr_galcat['y_hsc']
    arr_mags['zmag_org'] = arr_galcat['z_hsc']
    
    arr_mags['flux_OII'] = arr_galcat['flux_OII']
    arr_mags['SNR_OII_PFS'] = arr_galcat['SNR_PFS_OII']
    arr_mags['SNR_OII_DESI'] = arr_galcat['SNR_PFS_OII']/((900/1800)**0.5 * (8.2/4)**0.5 * (1/0.7)) # comment *1
    
    factor = 1/5 # comment *2
    sigmaf_gmag_HSC = 10**(-0.4*limmagSN5_g_HSC)/5 * arr_mags['R_e'] * factor
    sigmaf_rmag_HSC = 10**(-0.4*limmagSN5_r_HSC)/5 * arr_mags['R_e'] * factor
    sigmaf_imag_HSC = 10**(-0.4*limmagSN5_i_HSC)/5 * arr_mags['R_e'] * factor
    sigmaf_ymag_HSC = 10**(-0.4*limmagSN5_y_HSC)/5 * arr_mags['R_e'] * factor
    sigmaf_zmag_HSC = 10**(-0.4*limmagSN5_z_HSC)/5 * arr_mags['R_e'] * factor
    #print(sigmaf_gmag_HSC,sigmaf_rmag_HSC,sigmaf_imag_HSC,sigmaf_ymag_HSC,sigmaf_zmag_HSC)
    
    sigmaf_gmag_DEcaLS = 10**(-0.4*limmagSN5_g_DEcaLS)/5 * arr_mags['R_e'] * factor
    sigmaf_rmag_DEcaLS = 10**(-0.4*limmagSN5_r_DEcaLS)/5 * arr_mags['R_e'] * factor
    sigmaf_zmag_DEcaLS = 10**(-0.4*limmagSN5_z_DEcaLS)/5 * arr_mags['R_e'] * factor
    #print(sigmaf_gmag_DEcaLS,sigmaf_rmag_DEcaLS,sigmaf_zmag_DEcaLS)
    
    sigmaf_gmag_MzLS = 10**(-0.4*limmagSN5_g_MzLS)/5 * arr_mags['R_e'] * factor
    sigmaf_rmag_MzLS = 10**(-0.4*limmagSN5_r_MzLS)/5 * arr_mags['R_e'] * factor
    sigmaf_zmag_MzLS = 10**(-0.4*limmagSN5_z_MzLS)/5 * arr_mags['R_e'] * factor
    
    arr_mags['dgmag_HSC'] = -2.5*np.log10( 1 + np.random.normal(0, sigmaf_gmag_HSC, size=arr_mags.shape[0])/10**(-0.4*arr_mags['gmag_org']) )
    arr_mags['drmag_HSC'] = -2.5*np.log10( 1 + np.random.normal(0, sigmaf_rmag_HSC, size=arr_mags.shape[0])/10**(-0.4*arr_mags['rmag_org']) )
    arr_mags['dimag_HSC'] = -2.5*np.log10( 1 + np.random.normal(0, sigmaf_imag_HSC, size=arr_mags.shape[0])/10**(-0.4*arr_mags['imag_org']) )
    arr_mags['dymag_HSC'] = -2.5*np.log10( 1 + np.random.normal(0, sigmaf_ymag_HSC, size=arr_mags.shape[0])/10**(-0.4*arr_mags['ymag_org']) )
    arr_mags['dzmag_HSC'] = -2.5*np.log10( 1 + np.random.normal(0, sigmaf_zmag_HSC, size=arr_mags.shape[0])/10**(-0.4*arr_mags['zmag_org']) )
    
    arr_mags['dgmag_DEcaLS'] = -2.5*np.log10( 1 + np.random.normal(0, sigmaf_gmag_DEcaLS, size=arr_mags.shape[0])/10**(-0.4*arr_mags['gmag_org']) )
    arr_mags['drmag_DEcaLS'] = -2.5*np.log10( 1 + np.random.normal(0, sigmaf_rmag_DEcaLS, size=arr_mags.shape[0])/10**(-0.4*arr_mags['rmag_org']) )
    arr_mags['dzmag_DEcaLS'] = -2.5*np.log10( 1 + np.random.normal(0, sigmaf_zmag_DEcaLS, size=arr_mags.shape[0])/10**(-0.4*arr_mags['zmag_org']) )
    
    arr_mags['dgmag_MzLS'] = -2.5*np.log10( 1 + np.random.normal(0, sigmaf_gmag_MzLS, size=arr_mags.shape[0])/10**(-0.4*arr_mags['gmag_org']) )
    arr_mags['drmag_MzLS'] = -2.5*np.log10( 1 + np.random.normal(0, sigmaf_rmag_MzLS, size=arr_mags.shape[0])/10**(-0.4*arr_mags['rmag_org']) )
    arr_mags['dzmag_MzLS'] = -2.5*np.log10( 1 + np.random.normal(0, sigmaf_zmag_MzLS, size=arr_mags.shape[0])/10**(-0.4*arr_mags['zmag_org']) )
    
    # (S/N) is computed as (f + df)/sigma_f
    arr_mags['gflux_SN_HSC'] = 10**(-0.4*(arr_mags['gmag_org']+arr_mags['dgmag_HSC']))/sigmaf_gmag_HSC
    arr_mags['rflux_SN_HSC'] = 10**(-0.4*(arr_mags['rmag_org']+arr_mags['drmag_HSC']))/sigmaf_rmag_HSC
    arr_mags['iflux_SN_HSC'] = 10**(-0.4*(arr_mags['imag_org']+arr_mags['dimag_HSC']))/sigmaf_imag_HSC
    arr_mags['yflux_SN_HSC'] = 10**(-0.4*(arr_mags['ymag_org']+arr_mags['dymag_HSC']))/sigmaf_ymag_HSC
    arr_mags['zflux_SN_HSC'] = 10**(-0.4*(arr_mags['zmag_org']+arr_mags['dzmag_HSC']))/sigmaf_zmag_HSC
    
    arr_mags['gflux_SN_DEcaLS'] = 10**(-0.4*(arr_mags['gmag_org']+arr_mags['dgmag_DEcaLS']))/sigmaf_gmag_DEcaLS
    arr_mags['rflux_SN_DEcaLS'] = 10**(-0.4*(arr_mags['rmag_org']+arr_mags['drmag_DEcaLS']))/sigmaf_rmag_DEcaLS
    arr_mags['zflux_SN_DEcaLS'] = 10**(-0.4*(arr_mags['zmag_org']+arr_mags['dzmag_DEcaLS']))/sigmaf_zmag_DEcaLS
    
    arr_mags['gflux_SN_MzLS'] = 10**(-0.4*(arr_mags['gmag_org']+arr_mags['dgmag_MzLS']))/sigmaf_gmag_MzLS
    arr_mags['rflux_SN_MzLS'] = 10**(-0.4*(arr_mags['rmag_org']+arr_mags['drmag_MzLS']))/sigmaf_rmag_MzLS
    arr_mags['zflux_SN_MzLS'] = 10**(-0.4*(arr_mags['zmag_org']+arr_mags['dzmag_MzLS']))/sigmaf_zmag_MzLS
                                       
    return arr_mags
