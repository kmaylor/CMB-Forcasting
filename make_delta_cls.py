
import csv
from numpy import array, diag, sqrt, arange, dot
from Covariances.create_basic_cov import *
from Models.Planck.Planck_model import planck_model
from Models.SPT3G.spt3G_model import spt3G_model
import pickle as pk
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
                           'lensing':'yes',
                           }

#spt3g_cov=create_basic_cov(.035,beam_width,noise,model='classy',
#specrange = (lmin,max([T_lmax,P_lmax])),
#                               bin_size = 1,spectra=spectra,params=params)

res=pk.load(open('Covariances/Dls_residuals_spt3g_july_2018_with_1_f_noise.pkl','rb'),fix_imports=True,encoding='latin1')
ell = res['ILC'][0][:4990]

    
 
cmb=planck_model(model='classy',specrange = [('TT',(0,5001)),
                                ('EE',(0,5001)),
                                ('TE',(0,5001)),('BB',(0,5001)),]).get_cmb_fgs_and_cals(params)[0]
spt3g_cov = {}
spt3g_cov['BB'] = (2/((2*ell+1)*0.036))*(cmb['BB'][10:5000] + res['ILC'][2][:4990])**2
spt3g_cov['EE'] = (2/((2*ell+1)*0.036))*(cmb['EE'][10:5000] + res['ILC'][2][:4990])**2
spt3g_cov['TT'] = (2/((2*ell+1)*0.036))*(cmb['TT'][10:5000] + res['ILC'][1][:4990])**2  
spt3g_cov['TE'] = (1/((2*ell+1)*0.036))*(cmb['TE'][10:5000]**2 + res['ILC'][1][:4990]*res['ILC'][2][:4990])  
for k in cmb.keys():
    results = zip(arange(0,5001),cmb[k],sqrt(spt3g_cov[k]))
    with open('spt3g'+k+'_spectra_and_errors.csv','w') as f:
        writer = csv.writer(f,delimiter='\t')
        writer.writerows(results)
lmax=3005
lmin=5
fsky=0.035
m=spt3G_model(model='classy',specrange = [('PP',(lmin,lmax))],bin_size=10)
spec = dot(m.windows['PP'],m.get_cmb_fgs_and_cals(params)[0]['PP'][:lmax+1][m.windowrange['PP']])/(2.725*1e6)**2
lensing_recon_cov_file = 'Covariances/nlpp_ilc_res_SR_july2018_with_1_f_noise.pkl'
lensing_err = pk.load(open(lensing_recon_cov_file,'rb'),fix_imports=True,encoding='latin1')['NMV'][1:-1]
ell=pk.load(open(lensing_recon_cov_file,'rb'),fix_imports=True,encoding='latin1')['L'][1:-1]
PP_cov=((2/((2*ell+1)*fsky))*(lensing_err+spec)**2)/10
results = zip(ell,spec,sqrt(PP_cov))
with open('spt3gPP_spectra_and_errors.csv','w') as f:
        writer = csv.writer(f,delimiter='\t')
        writer.writerows(results)
