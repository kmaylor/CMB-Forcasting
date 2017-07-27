
from numpy import loadtxt, array, arange, log, dot, pi, exp, sqrt, zeros, floor, ceil, errstate,shape, hstack
from Models.CMB.Fisher_camb import camb_model
from Models.Foregrounds.planck_foregrounds import foregrounds
from collections import OrderedDict


class planck_model():
    
    def __init__(self,
                 specrange = [('TT',slice(2,3000)),
                                ('EE',slice(2,5000)),
                                ('TE',slice(2,5000))],
                 bin_size = 24):

        binn=bin_size
        self.order = [i[0] for i in specrange]
        self.windowrange = dict(specrange)
        
        self.windows =  {k:array([self.top_hat(l,v,binn) 
                               for l in range(v.start+int(binn/2),v.stop+1-
                                              int(binn/2),binn)]) for k,v in self.windowrange.items()}
        self.numbins = max([shape(self.windows[k])[0] for k in self.windows.keys()]) 
        self.cmb=camb_model()
        self.foregrounds=foregrounds()
        
    def __call__(self,**params):
        if 'As' not in params.keys(): params['As']=params.get('clamp',1.93e-09)*exp(2*params['tau'])
        if 'omch2' not in params.keys(): params['omch2']=params.get('ommh2',0.13)-params['ombh2']
        params['observations'] = self.windows.keys()
        cmb = self.cmb(**params)
        foregrounds = self.foregrounds(**params)
        cals = {'TT':1/params.get('Tcal',1)**2,
                'TE':1/(params.get('Tcal',1)**(2)*params.get('Pcal',1)),
                'EE':1/((params.get('Tcal',1)**(2)*params.get('Pcal',1)**(2)))}
        full_dbs=[]
        for k in self.order:
            dbs=zeros(self.numbins)
            dls = cals[k]*(cmb[k][self.windowrange[k]])#+foregrounds[k][self.windowrange[k]])
            dbs[:shape(self.windows[k])[0]] = dot(self.windows[k],dls)
            full_dbs=hstack([full_dbs,dbs])
        return full_dbs
    
    #dbs={}
    #    full_dbs=[]
    #    for k in self.order:
    #        dls = cals[k]*(cmb[k][self.windowrange[k]])#+foregrounds[k][self.windowrange[k]])
    #        dbs[k] = dot(self.windows[k],dls)
    #    for i in range(max(self.numbins.values())):
    #        for k in self.order:
    #            if i < self.numbins[k]:
    #                full_dbs.append(dbs[k])
    #    return full_dbs
    
    def top_hat(self,loc,win_range,binn):
        th = zeros(win_range.stop+1)
        th[int(loc-floor(binn/2.)):int(loc+ceil(binn/2.))]+=(1/binn)
        return th[win_range]