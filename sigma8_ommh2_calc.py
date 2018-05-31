import pickle as p
import pandas as pd
import cosmoslik as K
import camb
from numpy.random import multivariate_normal
from numpy import zeros,exp,mean,std
from numpy.linalg import inv

class sigma8_CAMB(object):
    def __init__(self):
        import camb as _camb
        self._camb=_camb
        
    def __call__(self,params):
        camb_params={'H0':params[0], 'ombh2':params[1], 'omch2':params[2]-params[1], 'ns':params[3],
                'As':params[4]*exp(2*params[5])*1e-9,'tau':params[5]}
        cp=self._camb.set_params(**camb_params)
        cp.set_matter_power(redshifts=[0], kmax=2.0)
        sigma8 = self._camb.get_results(cp).get_sigma8()
        return (sigma8*(cp.omegab+cp.omegac)**.25,sigma8*((cp.omegab+cp.omegac)/.3)**.5)
        
def planck_chain_means(param,model,experiment):
    planck_ex=experiment
    chain_path='/nfs/home/kmaylor/Official_Planck_chains/base'+model[4:]+'/plikHM'+planck_ex[6:]+ \
                '/base'+model[4:]+'_plikHM'+planck_ex[6:]
    if 'SPT3G' in experiment:
        if 'BAO' in experiment:
            try:
                return planck_chain_means(param,model,'Planck_TTTEEE_lowTEB_lensing_BAO')
            except Exception:
                return planck_chain_means(param,model,'Planck_TTTEEE_lowTEB')
        elif 'lensing' in experiment:
            try:
                return planck_chain_means(param,model,'Planck_TTTEEE_lowTEB_lensing')
            except Exception:
                return planck_chain_means(param,model,'Planck_TTTEEE_lowTEB')
        else:
            return planck_chain_means(param,model,'Planck_TTTEEE_lowTEB')
    elif '800' in experiment:
        return planck_chain_means(param,model,'Planck_TT_lowTEB')
    else:
        Planck_chain=K.chains.load_cosmomc_chain(chain_path).burnin(1000).join()
        Planck_chain=add_params(Planck_chain)
        return Planck_chain.mean(param)

def add_params(planck):
    '''Add parameters to planck chain'''
    planck['ommh2']=planck['omegabh2']+planck['omegach2']
    planck['H0']=planck['H0*']
    planck['clamp']=planck['clamp*']
    planck['ombh2']=planck['omegabh2']
    planck['omch2']=planck['omegach2']
    planck['YHe']=planck.get('yhe',planck['H0*'])
    planck['omk']=planck.get('omegak',planck['H0*'])
    planck['As']=planck['clamp']*exp(2*planck['tau'])*1e-9
    planck['cosmomc_theta']=planck['theta']/100
    return planck

def Fcov(model,ex,p=None,theta=False):
    if theta: 
        df= F_theta[model][ex]
    else: df=F[model][ex]
    if p!=None:
        return pd.DataFrame(inv(df.as_matrix()),index=df.columns,columns=df.columns)[p].loc[p]
    else:
        return pd.DataFrame(inv(df.as_matrix()),index=df.columns,columns=df.columns)

F=p.load(open('Saved_Fisher_Matrices\All_Planck_and_SPT3G_Fisher_matrices.p','rb'))
for k in F.keys():
    F[k]['Planck_TT_lmax_800'].at['tau','tau']+=.02**(-2)
    F[k]['Planck_TT_lmax_800_SPT3G_lensing_fsky_0.035'].at['tau','tau']+=.02**(-2)
    F[k]['SPT3G_lensing_fsky_0.035'].at['tau','tau']+=.02**(-2)

experiments=['Planck_TTTEEE_lowTEB',
             'Planck_TTTEEE_lowTEB_lensing',
             'Planck_TTTEEE_lowTEB_SPT3G_lensing_fsky_0.035',
             'Planck_TTTEEE_lowTEB_lensing_SPT3G_lensing_fsky_0.035',
             'Planck_TT_lmax_800_SPT3G_lensing_fsky_0.035',
             'SPT3G_lensing_fsky_0.035',
            'Planck_TT_lmax_800']
sigma8_ommh2=[() for i in experiments]
S8=[() for i in experiments]
sig8_camb=sigma8_CAMB()
for i,ex in enumerate(experiments):
    try:
        mu=planck_chain_means(['H0','ombh2','ommh2','ns','clamp','tau'],'lcdm',ex)
        sigma = Fcov('lcdm',ex,['H0','ombh2','ommh2','ns','clamp','tau'])
        sigma8_ommh2s,S8s=zip(*[sig8_camb(p) for p in multivariate_normal(mu,sigma,1000)])
        sigma8_ommh2[i]=(mean(sigma8_ommh2s),std(sigma8_ommh2s))
        S8[i]=(mean(S8s),std(S8s))
    except KeyError:
        continue
p.dump(sigma8_ommh2,open('Extra_data_for_plots/sigma8_ommh2_0.25_means_stds_fsky_0.035.pkl','wb'))
p.dump(S8,open('Extra_data_for_plots/S8_means_stds_fsky_0.035.pkl','wb'))
