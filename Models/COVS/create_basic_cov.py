
from Models.CMB.Fisher_camb import camb_model
from numpy import pi, exp, arange, outer, sqrt, array, zeros, floor, ceil, dot, ndarray, diag
from itertools import product
from pickle import dump


def top_hat(loc,win_range,binn):
    th = zeros(win_range.stop+1)
    th[int(loc-floor(binn/2.)):int(loc+ceil(binn/2.))]+=(1/binn)
    return th[win_range]

def calc_noise_weight(noise,units,fsky,time):
    if units == 'uks':
        w=lambda n:(n**2)*(4*pi*fsky)/time
        return {'TT':w(noise/sqrt(2)),'EE':w(noise),'TE':0.0}
    elif units == 'uk2sr':
        return {'TT':noise/2,'EE':noise,'TE':0.0}
    elif units == 'ukarc':
        w=lambda n:n**2/11818119.1464 #big number is the # of arcmin^2 per steradian
        return {'TT':w(noise/sqrt(2)),'EE':w(noise),'TE':0.0}
    else:
        raise ValueError("Noise units musk be either uks, uk2sr, or ukarc. See \
                          documentation for more details.")
 
def make_full_cov(spectra,covs):
    names = spectra
    onum = len(spectra)
    lnum = len(diag(list(covs.values())[0])) 
    olnum = onum*lnum
    full_cov = zeros([olnum,olnum])
    for i,k in enumerate(names):
        for j,l in enumerate(names):
            try:
                full_cov[i*lnum:(i+1)*lnum,j*lnum:(j+1)*lnum] = covs[k+l]
            except KeyError:
                full_cov[i*lnum:(i+1)*lnum,j*lnum:(j+1)*lnum] = covs[l+k]
    return full_cov


def create_basic_cov(fsky,
                     beam_FWHM,
                     filename,
                     params=None,
                     noise=(None,'uks'),
                     time=3e7,
                     windowrange = slice(650,5000),
                     bin_size = 24,
                     spectra=['TT','EE','TE'],
                     ):
    """
    fsky: Fraction of sky coverage
    beam_FWHM: beam_FWHM in radians
    params: dictionary of parameter to make fiducial model with CAMB
    noise: tuple of magnitude of noise and string specifying noise units, either
            'uks' = uKs^.5
            'uk2sr' = uK^2-steradians
            'ukarc' = uK-arcmin
            units will be converted to uK^2-steradians will add conversion to unitless in future
    time: observation time, only used if noise given in uKs^.5
    spectra: spectra to compute errors for
    """
    ### Fiducial Model for Covariance ###
    if params == None:
        params={'H0':None,'cosmomc_theta': 1.0409e-2,
                           'ombh2':0.0222,
                           'omch2':0.12,
                           'tau':0.078,
                           'As':1.881e-09*exp(2*0.068),
                           'ns':0.965,
                           'lmax':6000,
                           'lens_potential_accuracy':2.0}

    cmb=camb_model()(**params)
    #####################################
    
    ### NOISE TERMS ###
    w_inv = calc_noise_weight(noise[0],noise[1],fsky,time)
    B2_l_inv = lambda l: exp(l*(l+1)*(beam_FWHM/2.355)**2)
    ###################
    
    ### WINDOWS ###
    binn=bin_size
    windows =  array([top_hat(l,windowrange,binn) 
                    for l in range(windowrange.start+int(binn/2),windowrange.stop+1-int(binn/2),binn)])
    ###############
    
    ### MATRIC CALCULATION ###
    SNB = lambda k,ell: cmb[k] + w_inv[k]*B2_l_inv(ell)*ell*(ell+1)/(2*pi)

    covs = {}
    ell = arange(params['lmax']+1)
    
    for k1,k2 in product(spectra,spectra):
        if k1==k2:
            if k1 == 'TE':
                covs[k1+k2] = diag(dot(windows,((1/((2*ell+1)*fsky))*(SNB(k1,ell)**2 + 
                                                         SNB('TT',ell)*SNB('EE',ell)))[windowrange]))/(binn-1)
            else:
                covs[k1+k2] = diag(dot(windows,((2/((2*ell+1)*fsky))*SNB(k1,ell)**2)[windowrange]))/(binn-1)
        elif (k1 == 'TT' and k2 =='EE'):
            covs[k1+k2] = diag(dot(windows,((2/((2*ell+1)*fsky))*SNB('TE',ell)**2)[windowrange]))/(binn-1)
        elif ((k1 == 'TT' or k1=='EE') and k2 == 'TE'):
            covs[k1+k2] = diag(dot(windows,((2/((2*ell+1)*fsky))*SNB('TE',ell)*SNB(k1,ell))[windowrange]))/(binn-1)
    #########################
    
    ### SAVE RESULTS ###
    dump( covs, open( filename, "wb" ) )
