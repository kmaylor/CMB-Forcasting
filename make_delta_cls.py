
import csv
from numpy import array, diag, sqrt, arange
from Covariances.create_basic_cov import *
from Models.Planck.Planck_model import planck_model
beam_width=array([0.00032*150/90,.00032,.00032*150/220])
noise=[array([3.6,3.3,8.5]),'ukarc']
lmin=0
P_lmax=5001
T_lmax=5001
spectra=['TT','EE','TE','BB']

params={'H0':None,'cosmomc_theta': 1.04106e-2,
                           'ombh2':0.02227,
                           'omch2':0.1184,
                           'tau': 0.067,
                           'As':2.139e-09,
                           'ns': 0.9681,
                           'lmax':6000,
                           'DoLensing':True,
                           'lens_potential_accuracy':2.0,
                           'AccuracyBoost':2.0,
                           'lSampleBoost':2.0,
                           'lAccuracyBoost':2.0}

spt3g_cov=create_basic_cov(.06,beam_width,noise,model='camb',
specrange = (lmin,max([T_lmax,P_lmax])),
                               bin_size = 5,spectra=spectra,params=params)



    
 
cmb=planck_model(model='camb',specrange = [('TT',(0,3001)),
                                ('EE',(0,5001)),
                                ('TE',(0,5001)),('BB',(0,5001)),]).get_cmb_fgs_and_cals(params)[0]
                              
for k in cmb.keys():
    results = zip(arange(0,5001),cmb[k][2:],sqrt(diag(spt3g_cov[k+k])))
    with open('spt3g_800'+k+'_spectra_and_errors.csv','w') as f:
        writer = csv.writer(f,delimiter='\t')
        writer.writerows(results)
