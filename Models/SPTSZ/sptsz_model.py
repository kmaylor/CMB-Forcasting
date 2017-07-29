
from numpy import loadtxt, array, arange, log, dot, pi, exp, sqrt
from Models.CMB.Fisher_camb import camb_model
from Models.Foregrounds.spt_foregrounds import foregrounds

class sptsz_model():
    '''
    SPTSZ model for testing purposes, made some changes since last used, might be broken.
    '''
    
    def __init__(self):
        
        self.windows = array([loadtxt('/home/kmaylor/Python_Projects/PlanckVSPT/bandpowers/windows/window_lps12/window_%i'%i)[:,1] for i in range(1,48)])
        self.windowrange = (lambda x: slice(int(min(x)),int(max(x)+1)))(loadtxt('/home/kmaylor/Python_Projects/PlanckVSPT/bandpowers/windows/window_lps12/window_1')[:,0])
        self.cmb=camb_model()
        self.foregrounds=foregrounds()
        
    def __call__(self,**params):
        params['As']=params['clamp']*exp(2*params['tau'])
        params['omch2']=params['ommh2']-params['ombh2']
        self.dl = self.cmb(**params)['TT'][self.windowrange]+self.foregrounds(**params)['TT'][self.windowrange]
        self.dl *= (1+.26*1.23e-3*self.DlnClDlnl(self.dl))
        return dot(self.windows,self.dl)*params.get('cal',1)
    
    def DlnClDlnl(self,y):
        
        x = arange(self.windowrange.start,self.windowrange.stop)
        lnx=log(x)
        lny=log(y*2*pi/(x*(x+1)))
        return array([0]+[(lny[i+1]-lny[i])/(lnx[i+1]-lnx[i]) for i in arange(len(y)-1)])
    