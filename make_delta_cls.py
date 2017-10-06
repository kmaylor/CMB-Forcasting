
import csv
from numpy import array, diag, sqrt, arange
from Covariances.create_basic_cov import *
from Models.Planck.Planck_model import planck_model
beam_width=array([.0003])
noise=[array([0]),'ukarc']
lmin=2
P_lmax=800
T_lmax=800
spectra=['TT','EE','TE']
spt3g_cov=create_basic_cov(.7,beam_width,noise,model='camb',
specrange = (lmin,max([T_lmax,P_lmax])),
                               bin_size = 5,spectra=spectra)

params={'H0':None,'cosmomc_theta': 1.04106e-2,
                           'ombh2':0.02227,
                           'omch2':0.1184,
                           'tau': 0.067,
                           'As':2.139e-09,
                           'ns': 0.9681,
                           'lmax':6000,}
                           #'lens_potential_accuracy':2.0,
                           #'AccuracyBoost':2.0,
                           #'lSampleBoost':2.0,
                           #'lAccuracyBoost':2.0}

    
 
cmb=planck_model(model='camb').get_cmb_fgs_and_cals(params)[0]
#cmb=planck_model(model='camb',specrange = [('TT',(50,3000)),
#                                ('EE',(50,5000)),
#                                ('TE',(50,5000))])(**params)
for k in cmb.keys():
    results = zip(arange(2,5001),cmb[k][2:],sqrt(diag(spt3g_cov[k+k])))
    with open('planck_800'+k+'_spectra_and_errors.csv','w') as f:
        writer = csv.writer(f,delimiter='\t')
        writer.writerows(results)
