class model(): 
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
                 As=None,
                 H0=None,
                 cosmomc_theta = None,
                 k_eta_max_scalar=None,
                 lmax=6000,
                 massive_neutrinos=None,
                 massless_neutrinos=None,
                 mnu=None,
                 nnu=None,
                 NonLinear=None,
                 ns=None,
                 ombh2=None,
                 omch2=None,
                 omk=None,
                 pivot_scalar=None,
                 tau=None,
                 YHe=None,
                 AccuracyBoost=None,
                 lens_potential_accuracy=None,
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
        
        #feed params to camb and calulate power spectra, returned in specs
        cp=self._camb.set_params(**params)
        self.result = self._camb.get_results(cp)
        specs = self.result.get_cmb_power_spectra(spectra=['unlensed_total','lens_potential'])
        
        #Sort spectra by component instead of ell. Easier for adding foregrounds
        #need to fix units for lensing             
        tmp= dict(list(zip(['TT','EE','BB','TE','PP','PT','PE'],
                    [ x for y in ['unlensed_total','lens_potential'] 
                     for x in (cp.TCMB*1e6)**2*specs[y].T] )))
        if spectra == None:
            return tmp
        else:
            return {k:v for k,v in tmp.items() if (k in spectra)} #grab the components we want in spectra
        
