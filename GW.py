""" Physics functions as well as constants.

In the GW computation we choose units m, eV, and s.  This script converts units from darkhistory.physics to be used for these computations.

"""

import pickle
import sys

import numpy as np
from scipy.interpolate import interp1d
from scipy.special import zeta
import darkhistory.physics as phys

# from config import data_path
from config import load_data

#############################################################################
#############################################################################
# Fundamental Particles and Constants                                       #
#############################################################################
#############################################################################

mp          = phys.mp
"""Proton mass in eV."""
me          = phys.me
"""Electron mass in eV."""
mHe         = phys.mHe
"""Helium nucleus mass in eV."""
hbar        = phys.hbar
"""hbar in eV s."""
c           = phys.c/100
"""Speed of light in m s\ :sup:`-1`\ ."""
kB          = phys.kB
"""Boltzmann constant in eV K\ :sup:`-1`\ ."""
alpha       = phys.alpha
"""Fine structure constant."""
ele = np.sqrt(4*np.pi*phys.alpha)
"""Electron charge (Lorentz-Heaviside units)."""
G = phys.G*1e-3*1.6e-19/c**2
"""Newton's Gravitational Constant in m\ :sup:`3` eV\ :sup:`-1` s\ :sup:`-2`\ ."""

mass = {
    'e': me,         'mu': 105.6583745e6, 'tau': 1776.86e6,
    'c': 1.275e9  ,  'b' :   4.18e9,        't': 173.1e9,
    'W': 80.379e9  , 'Z' :  91.1876e9,      'h': 125.18e9
}
"""Masses of Standard Model particles."""
thomson_xsec = phys.thomson_xsec/100**2
"""Thomson cross section in m\ :sup:`2`\ ."""
stefboltz    = phys.stefboltz*100**2
"""Stefan-Boltzmann constant in eV\ :sup:`-3` m\ :sup:`-2` s\ :sup:`-1`\ .
"""
ele_rad      = phys.ele_rad/100
"""Classical electron radius in m."""
ele_compton  = phys.ele_compton/100
"""Electron Compton wavelength in m."""

#############################################################################
#############################################################################
# Cosmology                                                                 #
#############################################################################
#############################################################################

#########################################
# Densities and Hubble                  #
#########################################

h    = phys.h
""" h parameter."""
H0   = phys.H0*phys.hbar
""" Hubble parameter today in eV."""

omega_m      = phys.omega_m
""" Omega of all matter today."""
omega_rad    = phys.omega_rad
""" Omega of radiation today."""
omega_lambda = phys.omega_lambda
""" Omega of dark energy today."""
omega_baryon = phys.omega_baryon
""" Omega of baryons today."""
omega_DM      = phys.omega_DM
""" Omega of dark matter today."""
rho_crit     = phys.rho_crit*100**3
""" Critical density of the universe in eV m\ :sup:`-3`\ . 

See [1] for the definition. This is a mass density, with mass measured in eV.
"""
rho_DM       = phys.rho_DM*100**3
""" DM density in eV m\ :sup:`-3`\ ."""
rho_baryon   = phys.rho_baryon*100**3
""" Baryon density in eV m\ :sup:`-3`\ ."""
nB          = phys.nB*100**3
""" Baryon number density in eV m\ :sup:`-3`\ ."""

YHe         = phys.nHe*phys.mHe/(phys.nHe*phys.mHe+phys.nH*phys.mp)
"""Helium abundance by mass."""
nH          = phys.nH*100**3
""" Atomic hydrogen number density in m\ :sup:`-3`\ ."""
nHe         = phys.nHe*100**3
""" Atomic helium number density in m\ :sup:`-3`\ ."""
nA          = phys.nA*100**3
""" Hydrogen and helium number density in m\ :sup:`-3`\ .""" 
chi         = phys.chi
"""Ratio of helium to hydrogen nuclei."""


kappa = (16*np.pi*G)**(1/2)/c**2 
""" Domcke and Garcia-Cely constant kappa in s eV\ :sup:`-1/2`\ m\ :sup:`-1/2`\ """
kappa_converted = kappa/np.sqrt(phys.hbar*c)*0.0195592 
""" Domcke and Garcia-Cely constant kappa in s eV m\ :sup:`-1`\ Gauss\ :sup:`-1`\ (so that magnetic field can be supplied in Gauss)"""


def hubble(z):
    """ Hubble parameter in eV .

    Assumes a flat universe.

    Parameters
    ----------
    z : float
        The z value (rs-1) of interest.

    Returns
    -------
    float
    """
    return phys.hubble(1+z)*phys.hbar
    
def rho_c(z):  
	""" Critical density as a function of redshift in eV/m^3.

    Parameters
    ----------
    z : float
        The z value (rs-1) of interest.

    Returns
    -------
    float
    """
	return 3*hubble(z)**2/(8*np.pi*G)/phys.hbar**2

def nH(z):
    """ Hydrogen number density as a function of redshift in m^-3.

    Parameters
    ----------
    z : float
        The z value (rs-1) of interest.

    Returns
    -------
    float
    """
    return (phys.omega_baryon*3*(hubble(0)/phys.hbar)**2/(8*np.pi*G))*(1+z)**3/(phys.mp+(phys.mHe*YHe)/(4*(1-YHe)))

def ne(xe,z):
    """ Electron number density as a function of redshift in m^-3.

    Parameters
    ----------
    xe : float
    	The free electron fraction.
    z : float
        The z value (rs-1) of interest.

    Returns
    -------
    float
    """
    return xe*nH(z)

def nHe(z):
    """ Helium number density as a function of redshift in m^-3.

    Parameters
    ----------
    z : float
        The z value (rs-1) of interest.

    Returns
    -------
    float
    """
    return (phys.omega_baryon*rho_c(0)*(1+z)**3 - phys.mp*nH(z))/phys.mHe

def chi(z):
    """ Ratio of helium to hydrogen nuclei as a function of redshift in m^-3.

    Parameters
    ----------
    z : float
        The z value (rs-1) of interest.

    Returns
    -------
    float
    """
    return nHe(z)/nH(z)
    
def plasma_freq(xe,z): # this turns out to be a bottleneck, idk if there's a way to speed it up?
    """ Plasma frequency as a function of redshift in eV.

    Parameters
    ----------
    xe : float
    	The free electron fraction.
    z : float
        The z value (rs-1) of interest.

    Returns
    -------
    float
    """
    return np.sqrt(ele**2*ne(xe,z)*(phys.hbar*c)**(3)/phys.me)

def redshifted_B(B0,z):
	""" Magnetic field as a function of redshift (units match input units).

    Parameters
    ----------
    B0 : float
    	Initial value of magnetic field (Gauss preferred).
    z : float
        The z value (rs-1) of interest.

    Returns
    -------
    float
    """
	return B0*(1+z)**2
    
def mu(x,xe,z): # this turns out to be a bottleneck, idk if there's a way to speed it up?
    """ Dimensionless parameter mu (see Domcke and Garcia-Cely).

    Parameters
    ----------
    x : float
    	Dimensionless frequency omega/T.
    xe : float
    	The free electron fraction.
    z : float
        The z value (rs-1) of interest.

    Returns
    -------
    float
    """
    if 1 - np.sqrt(1-plasma_freq(xe,z)**2/(x*phys.TCMB(1)*(1+z))**2) >= 1e-13:
        return np.sqrt(1-plasma_freq(xe,z)**2/(x*phys.TCMB(1)*(1+z))**2)
    else:
        return 1-plasma_freq(xe,z)**2/(2.*(x*phys.TCMB(1)*(1+z))**2) # Taylor expansion (not strictly necessary here)

def losc(x,B0,xe,z): # B0 must be in Gauss
    # returns a length in units m/s/eV; divide by phys.hbar to get a length in m
    """ Oscillation length (see Domcke and Garcia-Cely) in m/s/eV.

    Parameters
    ----------
    x : float
    	Dimensionless frequency omega/T.
    B0 : float
    	Initial value of magnetic field in Gauss.
    xe : float
    	The free electron fraction.
    z : float
        The z value (rs-1) of interest.

    Returns
    -------
    float
    """
    if 1 - np.sqrt(1-plasma_freq(xe,z)**2/(x*phys.TCMB(1)*(1+z))**2) >= 1e-13:
        return (2./np.sqrt(((x*phys.TCMB(1)*(1+z))**2*(1-mu(x,xe,z))**2)/c**2 + kappa_converted**2*redshifted_B(B0,z)**2*(1+z)**2))
    else:
        return (2./np.sqrt(((x*phys.TCMB(1)*(1+z))**2*(plasma_freq(xe,z)**2/(2.*(x*phys.TCMB(1)*(1+z))**2))**2)/c**2 + kappa_converted**2*redshifted_B(B0,z)**2*(1+z)**2))
        # Taylor expand 1 - mu if mu is too close to 1
        
def delta_l(z,delta_l0=3.0857e22):
	""" Delta l (see Domcke and Garcia-Cely) as a function of redshift (units match input units).

    Parameters
    ----------
    z : float
        The z value (rs-1) of interest.
    delta_l0 : float (default: 3.0857e22 m (=1Mpc))
    	Initial value of delta l (m preferred).

    Returns
    -------
    float
    """
	return delta_l0/(1+z)

def conversion_rate(x,B0,xe,z,delta_l0=3.0857e22): # B must be in Gauss, delta_l in cm
    # returns a value in 1/s.  Multiply by phys.hbar to get a value in eV
    """ Oscillation length (see Domcke and Garcia-Cely) in m/s/eV.

    Parameters
    ----------
    x : float
    	Dimensionless frequency omega/T.
    B0 : float
    	Initial value of magnetic field in Gauss.
    xe : float
    	The free electron fraction.
    z : float
        The z value (rs-1) of interest.
    delta_l0 : float (default: 3.0857e22 m (=1Mpc))
    	Initial value of delta l in meters.

    Returns
    -------
    float
    """
    K12 = -1j*np.sqrt(mu(x,xe,z))*kappa_converted*redshifted_B(B0,z)/(1+mu(x,xe,z)) # eV s/m
    return c*np.abs(K12)**2*losc(x,B0,xe,z)**2/(2*delta_l(z,delta_l0))

def particle_density(xe,z):
    """ Total particle number density as a function of redshift in m^-3.

    Parameters
    ----------
    xe : float
    	The free electron fraction.
    z : float
        The z value (rs-1) of interest.

    Returns
    -------
    float
    """
    return nH(z)*(1+xe+chi(z))