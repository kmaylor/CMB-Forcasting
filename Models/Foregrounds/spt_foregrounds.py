
from numpy import hstack, loadtxt, arange

class foregrounds():
    
    def __init__(self):
        self.sz_template = hstack([[0],loadtxt("/home/kmaylor/Python_Projects/PlanckVSPT/foreground_templates/SZ_template.txt")[:,1]])
        self.poisson_template = hstack([[0],loadtxt("/home/kmaylor/Python_Projects/PlanckVSPT/foreground_templates/poisson_template.txt")[:,1]])
        self.cluster_template = hstack([[0,0],loadtxt("/home/kmaylor/Python_Projects/PlanckVSPT/foreground_templates/cluster_template.txt")[:,1]])


    def __call__(self,
                 Asz=5.5,
                 Aps=19.3,
                 Acib=5,
                 **kwargs):
    
            ell = arange(len(self.sz_template))
            TT_foregrounds = Asz * self.sz_template + Aps * self.poisson_template + Acib * self.cluster_template
            
            
            return {'TT':TT_foregrounds}