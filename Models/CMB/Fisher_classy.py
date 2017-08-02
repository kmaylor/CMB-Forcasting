from numpy import arange, pi


class classy_model():
    """
    Compute the CMB power spectrum with CLASS.
    Based on work by: Brent Follin, Teresa Hamill, Marius Millea
    """

    #{cosmoslik name : class name}
    name_mapping = {
        'As':'A_s',
        'lmax':'l_max_scalars',
        'mnu':'m_ncdm',
        'nnu':'N_ncdm',
        'ns':'n_s',
        'nt':'n_t',
        'ombh2':'omega_b',
        'omch2':'omega_cdm',
        'omk':'Omega_k',
        'pivot_scalar':'k_pivot',
        'r':'r',
        'tau':'tau_reio',
        'Tcmb':'T_cmb',
        'Yp':'YHe',
    }


    def __init__(self):
        from classy import Class
        self.model = Class()


    def convert_params(self,**params):
        """
        Convert from CosmoSlik params to CLASS
        """
        params = {self.name_mapping.get(k,k):v for k,v in params.items()}
        if 'theta' in params:
            params['100*theta_s'] = 100*params.pop('theta') 
        return params
        
        
    def __call__(self,
                 As=None,
                 H0=None,
                 lmax=None,
                 mnu=None,
                 N_ncdm=None,
                 nrun=None,
                 ns=None,
                 ombh2=None, 
                 omch2=None,
                 omk=None,
                 output='tCl, lCl, pCl',
                 pivot_scalar=None,
                 r=None,
                 tau=None,
                 Tcmb=2.7255,
                 theta=None,
                 w=None,
                 Yp=None,
                 lensing='yes',
                 nokwargs =True,
                 spectra=None,
                 **kwargs):
        
        args=locals() #grabs all of the variables fed into __call__
        args.pop('self',[])
        
        if nokwargs: #Decide whether or not to add params that did not have keyword to CAMB call
            args.pop('kwargs',[])
        else:
            args.update(args.pop('kwargs',[]))
            
        args.pop('nokwargs',[]) 
        args.pop('spectra',[])
       
        #Remove args with None value, CAMB will use default instead
        params = {k:v for k,v in args.items() 
                         if v is not None}
        if params.get('cosmomc_theta',None) is not None:
            params['H0']=None
            
        self.model.set(self.convert_params(**params))
        self.model.compute()

        lmax = params['lmax']
        ell = arange(lmax+1)
        self.cmb_result = {x:(self.model.lensed_cl(lmax)[x.lower()])*Tcmb**2*1e12*ell*(ell+1)/2/pi
                           for x in ['TT','TE','EE','BB','PP','TP']}

        self.model.struct_cleanup()
        self.model.empty()
        return self.cmb_result
    
        if spectra == None:
            return self.cmb_result
        else:
            return {k:v for k,v in self.cmb_result.items() if (k in spectra)} #grab the components we want in spectra