
from numpy import pi, exp, arange, outer, sqrt, array, zeros, floor, ceil, dot, ndarray, diag
from itertools import product
from pickle import dump
from collections import Iterable
from Models.Planck.Planck_model import planck_model

#When using from create_basic_cov import * only the following line will be imported
__all__ = ['create_basic_cov','make_full_cov']

"""
This contains the code needed to construct a covariance matrix for the planck or SPT3G models.
First call create_basic_cov to create (and save) the block matrix for each spectra, TTTT etc., then 
use make_full_cov to convert this into the format needed to calculate the fisher matrix.
"""
def top_hat(loc,win_range,binn):
    '''
        Returns a top hat function centered on loc with range= win_range and bin size = binn.
    '''
    th = zeros(win_range.stop+1)
    th[int(loc-binn//2):int(loc+binn//2+1)]+=(1/binn)
    return th[win_range]

def convert_noise_weight(noise,units,fsky,time):
    '''
    Convert the input noise to units of uK-sr. Making the element for 'TE' zero allows for
    simplification of the code and the number of unique equations for calculating the block
    covariances is reduced to 4
    '''
    if units == 'uks':
        con=lambda n:time/((n**2)*(4*pi*fsky))
        return {'TT':con(noise),'EE':con(noise*sqrt(2)),'BB':con(noise*sqrt(2))}
    elif units == 'uk2sr':
        return {'TT':1/noise,'EE':1/noise*2,'BB':1/noise*2}
    elif units == 'ukarc':
        arcmin_per_sr = 11818113.9613 #big number is the # of arcmin^2 per steradian
        con=lambda n:arcmin_per_sr/n**2
        return {'TT':con(noise),'EE':con(noise*sqrt(2)),'BB':con(noise*sqrt(2))}
    else:
        raise ValueError("Noise units musk be either uks, uk2sr, or ukarc. See \
                          documentation for more details.")
 
def make_full_cov(spectra,covs):
    '''
    Covert the output from create_basic_cov to a block covariance matrix ordered by spectra.
    Example: if spectra = ['TT','EE'] then the output is a block matrix of the form
        |      |      |
        | TTTT | TTEE |
        |______|______|
        |      |      |
        | EETT | EEEE |
        |      |      |
    '''
    snum = len(spectra) #number of blocks
    lnum = len(diag(list(covs.values())[0])) #number of bandpowers 
    slnum = snum*lnum #size of full_cov = #block*#bandpowers
    full_cov = zeros([slnum,slnum])
    for i,k in enumerate(spectra):
        for j,l in enumerate(spectra):
            try:
                full_cov[i*lnum:(i+1)*lnum,j*lnum:(j+1)*lnum] = covs[k+l]
            except KeyError:
                full_cov[i*lnum:(i+1)*lnum,j*lnum:(j+1)*lnum] = covs[l+k]
    return full_cov


def create_basic_cov(fsky,
                     beam_FWHM,
                     noise,
                     time=3e7,
                     model='camb',
                     specrange = (2,5000),
                     bin_size = 24,
                     spectra=['TT','EE','TE'],
                     params=None,
                     filename=None,

                     ):
    """
    fsky: Fraction of sky coverage
    beam_FWHM: beam_FWHM in radians
    noise: list of magnitude of noise (expected in temperature) and string specifying noise units, either
            'uks' = uKs^.5
            'uk2sr' = uK^2-steradians
            'ukarc' = uK-arcmin
            units will be converted to uK^2-steradians will add conversion to unitless in future
    time: observation time, only used if noise given in uKs^.5
    specrange: the range of ells to be included in the covariance
    bin_size: the size of the bins for the covariance
    spectra: spectra to compute errors for
    params: dictionary of parameters to make fiducial model with CAMB, currently set to planck values
    filename: name of file to store each block matrix

    """
    ### Fiducial Model for Covariance ###
    if params == None:
        params={'H0':None,'cosmomc_theta': 1.04106e-2,
                           'ombh2':0.02227,
                           'omch2':0.1184,
                           'tau': 0.067,
                           'As':2.139e-09,
                           'ns': 0.9681,
                           'lmax':6000,}
                           #'lens_potential_accuracy':2.0}

    

    cmb=planck_model(model=model,specrange=[(k,specrange) for k in spectra]).get_cmb_fgs_and_cals(params)[0]

    #####################################
    
    ### NOISE TERMS ###
    if not isinstance(noise[0],Iterable): noise[0]=[noise[0]]
    w = convert_noise_weight(array(noise[0]),noise[1],fsky,time)
    B2_l = lambda l: [exp(-l*(l+1)*(b/2.355)**2) for b in beam_FWHM]
    ###################
    
    ### WINDOWS ###
    binn=bin_size
    windowrange=slice(specrange[0],specrange[1])
    windows =  array([top_hat(l,windowrange,binn) 
                    for l in range(windowrange.start+binn//2,windowrange.stop+1-binn//2,binn)])/(binn-1)
    ###############
    
    ### BLOCK MATRIX CALCULATION ###
    '''
    The output of camb (cmb) is in Dl so we multiply the noise term by ell(ell+1)/2pi
    '''
    
    def SNB(k,ell):
        try:
            return cmb[k] + (ell*(ell+1)/(2*pi))/dot(w[k],B2_l(ell))
        except KeyError:
            return cmb[k]

    covs = {}
    ell = arange(params['lmax']+1)
    
    # run through all block matrix calculations, the full matrix is symmetric so we don't need to do all of them.
    for k1,k2 in product(spectra,spectra):
        
        if k1==k2:
            if k1 == 'TE': #TETE block
                covs[k1+k2] = diag(dot(windows,((1/((2*ell+1)*fsky))*(SNB(k1,ell)**2 + 
                                                         SNB('TT',ell)*SNB('EE',ell)))[windowrange]))
                
            else: #TTTT,EEEE,BBBB blocks
                covs[k1+k2] = diag(dot(windows,((2/((2*ell+1)*fsky))*SNB(k1,ell)**2)[windowrange]))
                
        elif (k1 == 'TT' and k2 =='EE'): #TTEE block
            covs[k1+k2] = diag(dot(windows,((2/((2*ell+1)*fsky))*SNB('TE',ell)**2)[windowrange]))
            
        elif ((k1 == 'TT' or k1=='EE') and k2 == 'TE'): #TTTE and EETE blocks
            covs[k1+k2] = diag(dot(windows,((2/((2*ell+1)*fsky))*SNB('TE',ell)*SNB(k1,ell))[windowrange]))
            
    #########################
    
    ### SAVE RESULTS ###
    if filename !=None:
        # Save just the blocks to reduce file size since the full covariance can be made from them
        dump( covs, open( filename, "wb" ) )
    else:
        return covs
