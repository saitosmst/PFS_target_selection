#! /usr/bin/env python3
# -*- coding:utf-8 -*-

r"""
    author: SHUN SAITO (Missouri S&T)

    Time-stamp: <Wed Sep 23 10:18:14 CDT 2020>

    This code is developed for the target-selection purpose 
    in the PFS cosmology working group.


    This file provides useful functions to output summary statistics 
    for the PFS target selection.

    Note that the code is highly relying on the EL-COSMOS catalog 
    (Saito et al. 2020) which predicts the [OII] fluxes
    as well as broadband photometries from HSC, (g, r, i, y, z).

    Version history:
        0.0    initial migration from jupyter notebooks
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



def calc_target_stats(cat_input, selection, snr_threshold=6, snr_label='SNR_PFS_OII',
                      zlow=0.6, zhigh=2.4, verbose=0):
    """
    This is the main function to compute the summary statistics for the target selection. 
    """
    cat_tgt = cat_input[selection]
    num_tgt_perFoV = int(cat_tgt.shape[0]*AREA_PFSFoV/AREA_CMC)
    
    # assume that the fiber assignment follows the Poisson statistics
    mu = float(num_tgt_perFoV)/float(NUM_FIBER)
    num_fiber_assigned = int(NUM_FIBER*(1. - poisson.pmf(0., mu) + 1.
                                        - poisson.pmf(0., mu) - poisson.pmf(1., mu)))
    num_fiber_missed = 2*NUM_FIBER - num_fiber_assigned
    
    ######
    select_det = cat_tgt[snr_label] > snr_threshold
    select_zred = (cat_tgt['z_photo'] > zlow) & (cat_tgt['z_photo'] < zhigh)
    cat_pfs = cat_tgt[select_det & select_zred]
    f_success = np.float(cat_pfs.shape[0])/np.float(cat_tgt.shape[0])
    num_elg  = int(f_success*num_fiber_assigned)

    if verbose >= 1:
        print('***** Target selection statistics summary *****\n')
        print('Your threthold: {0} > {1}\n'.format(snr_label,snr_threshold))
        print('(1) # of fibers for which a galaxy is assigned: {0}'.format(num_fiber_assigned))
        print('# of fibers which miss a galaxy assignment: {0}'.format(num_fiber_missed))
        print('efficiency = (1)/(# of total fibers, {0}): {1:5.2f}%\n'.format(2*NUM_FIBER, 100.*(1. - poisson.pmf(0., mu) + 1.
                                                                               - poisson.pmf(0., mu) - poisson.pmf(1., mu))/2.))
        print('(2) # of target galaxies per FoV: {0}'.format(num_tgt_perFoV))
        print('completeness = (1)/(2): {0:5.2f}%\n'.format(100*num_fiber_assigned/num_tgt_perFoV))
        print('(3) # of detectable PFS ELGs per two visits per FoV: {0}'.format(num_elg))
        print('success rate = (3)/(1) = {0:5.2f}%\n'.format(100.*num_elg/num_fiber_assigned))
        if verbose >=2:
            print('effective factor: ', num_fiber_assigned/np.float(cat_tgt.shape[0]))
            print(' -> this factor should be multiplied to *cat_pfs* to get # of detectable galaxies per PFS FoV\n')
        print('***********************************************\n')

    return num_elg, cat_pfs, num_fiber_assigned/np.float(cat_tgt.shape[0])




def calc_dndz(arr_tmp, fac_eff, verbose=0):
    """
    Compute redshift distribution of the input catalog ('arr_tmp') following the binning in Takada+(2014)
    """

    dndzs = np.zeros(shape=(7,4))
    
    for iz in range(7):
        if iz<=4:
            zmin = 0.6 + iz*0.2
            zmax = 0.8 + iz*0.2
        else:
            zmin = 1.6 + (iz-5)*0.4
            zmax = 2.0 + (iz-5)*0.4
    
        Vs = integrate.quad(dVc_dzdOmega, zmin, zmax)[0]*AREA_PFS*(u.deg**2).to(u.sr)/1.e9 #[(Gpc/h)^3]
        select = (arr_tmp['z_photo']>=zmin) & (arr_tmp['z_photo']<zmax)
        Ng_FoV = int(arr_tmp[select].shape[0]*fac_eff)
        ng = Ng_FoV*AREA_PFS/AREA_PFSFoV/Vs/10.**5

        dndzs[iz,0] = (zmin+zmax)/2
        dndzs[iz,1] = Vs
        dndzs[iz,2] = Ng_FoV
        dndzs[iz,3] = ng

        if verbose >= 1:
            if iz==0:
                print('dn/dz: ')
            print('#{0}, {1:2.1f} <= z < {2:2.1f}, Vs[(Gpc/h)^3] = {3:4.3f}, Ng/FoV = {4}, ng[10^-4(h/Mpc)^3] = {5:4.3f}'.format(iz,zmin,zmax,Vs,Ng_FoV,ng))
    return dndzs


if __name__ == "__main__":
    #example code to use this function.
    arr_cmc = fits.open('/Volumes/ThunderBay4-40TB/CosmoTreasure/mock/cosmos/cmc_lam/combinedcatalog_flux_SNR-cos_full_phy_fz-a0.20_OI15config_added-Re.fits')[1].data
    
    # selection in (g, g-r, r-i)
    select_g  = (arr_cmc['g_hsc'] > 23.2) & (arr_cmc['g_hsc'] < 24.2)
    select_gr = ((arr_cmc['g_hsc']-arr_cmc['r_hsc'] > 0.05) 
                 & (arr_cmc['g_hsc']-arr_cmc['r_hsc'] < 0.35))
    select_ri = ((arr_cmc['g_hsc'] > 23.6) 
                 & (arr_cmc['r_hsc']-arr_cmc['i_hsc'] > 0.3))
    select_g_gr = select_g & select_gr
    cut_target  = select_g_gr & ~select_ri
    
    numELGs, pfs_mT14_sn6, fac_eff = calc_target_stats(arr_cmc, cut_target, snr_threshold=6, verbose=1)
    calc_dndz(pfs_mT14_sn6, fac_eff, verbose=1)
