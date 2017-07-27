class camb_model(): 
    """
    Make a call to an instance of this class to calculate the a CMB spectrum using CAMB.
    Unless specified otherwise it returns TT, EE, BB, and TE.
    
    Based on code from Cosmoslik by Marius Millea
    https://github.com/marius311/cosmoslik/blob/master/cosmoslik_plugins/models/camb.py
    """
    
    def __init__(self):
        
        import camb as _camb
        self._camb = _camb

    
    def __call__(self,
                 ALens=None,
                 As=2.1968681258131772e-09,
                 DoLensing=None,
                 H0=None,
                 cosmomc_theta = None,
                 k_eta_max_scalar=None,
                 lmax=2000,
                 massive_neutrinos=None,
                 massless_neutrinos=None,
                 mnu=None,
                 nnu=None,
                 NonLinear=None,
                 ns=0.92228442,
                 ombh2=0.02321343,
                 omch2=0.10740280,
                 omk=None,
                 pivot_scalar=None,
                 tau=0.0646503,
                 YHe=None,
                 AccuracyBoost=None,
                 lens_potential_accuracy=None,
                 nokwargs =True,
                 observations=None,
                 **kwargs):
        
        args=locals()
        args.pop('self',[])
        
        if nokwargs:
            args.pop('kwargs',[])
        else:
            args.update(args.pop('kwargs',[]))
            
        args.pop('nokwargs',[]) 
        args.pop('observations',[])
        
        params = {k:v for k,v in args.items() 
                         if v is not None}
        if params.get('cosmomc_theta',None) is not None:
            params['H0']=None

        cp=self._camb.set_params(**params)
        self.result = self._camb.get_results(cp)
        specs = self.result.get_cmb_power_spectra(spectra=['total','lens_potential'])
        #need to fix units for lensing             
        tmp= dict(list(zip(['TT','EE','BB','TE','PP','PT','PE'],
                    [ x for y in ['total','lens_potential'] 
                     for x in (cp.TCMB*1e6)**2*specs[y].T] )))
        if observations == None:
            return tmp
        else:
            return {k:v for k,v in tmp.items() if (k in observations)}
        