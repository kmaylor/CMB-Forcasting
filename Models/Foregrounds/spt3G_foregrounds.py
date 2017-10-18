
from numpy import hstack, loadtxt, arange, errstate, zeros
import os.path as osp


class foregrounds():
    '''
    Model for SPT3G foregrounds, uses SPTsz TT foregrounds
    '''
    
    def __init__(self):
        self.sz_template = hstack([[0],loadtxt(osp.join(osp.dirname(__file__),"spt3G_foreground_templates/SZ_template.txt"))[:,1]])
        self.poisson_template = hstack([[0],loadtxt(osp.join(osp.dirname(__file__),"spt3G_foreground_templates/poisson_template.txt"))[:,1]])
        self.cluster_template = hstack([[0,0],loadtxt(osp.join(osp.dirname(__file__),"spt3G_foreground_templates/cluster_template.txt"))[:,1]])

    def __call__(self,
                 Asz=5.5,
                 Aps=19.3,
                 Acib=5,
                 A_TEps = .16,
                 #A_EE80 = 2,
                 #a_TE = -2.42,
                 #a_EE = -2.42,
                 #D_PSEE_3000 = 2.5,
                 **kwargs):
    
            ell = arange(len(self.sz_template))
            TT_foregrounds = Asz * self.sz_template + Aps * self.poisson_template + Acib * self.cluster_template
            with errstate(divide='ignore'):
                EE_foregrounds = zeros(len(ell))#A_EE80*(ell/80.)**(a_EE+2) + D_PSEE_3000*(ell/3000.)**2
                TE_foregrounds = A_TEps*(ell/3000)**2
            
            return {'TT':TT_foregrounds, 'EE': EE_foregrounds, 'TE':TE_foregrounds}