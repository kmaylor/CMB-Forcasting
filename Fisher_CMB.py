
from copy import deepcopy
from numpy import dot, array, zeros, diag, sqrt, arange, shape
from scipy.linalg import cho_factor, cho_solve, inv
from collections import OrderedDict

__all__ = ['fish_param','Fisher_Matrix']

class fish_param(object):
    
    def __init__(self,base,step,prior=None):
        self.base = base
        self.step = step
        self.prior = prior


def derivative(model,params):
    '''
    Calculate the partial derivatives of a model with respect to the parameters to be included in the 
    Fisher matrix
    using finite differences. 
    '''
    dydx=[]
    base={l:m.base for l,m in params.items() if  isinstance(m,fish_param)}
    for k,v in params.items():
        if isinstance(v,fish_param):
            tmp = deepcopy(params)
            tmp.update(base)
            tmp[k]+=v.step/2.
            modelu = model(**tmp)
            tmp[k]-=v.step
            modeld = model(**tmp)
            dydx.append((modelu-modeld)/v.step) 
    return array(dydx)


def Fisher_Calc(model_der,cov):
    """
    Calculate the Fisher Matrix for a given model and covariance.
    """
    cho_cov = cho_factor(cov)
    return dot(model_der,cho_solve(cho_cov,model_der.T))


def add_priors(params,Fisher_matrix):
    priors = [ m.prior for l,m in params.items() if isinstance(m,fish_param)]
    for i,p in enumerate(priors):
        if p is not None:
            Fisher_matrix[i,i]+=p**(-2)
    return Fisher_matrix
            
            
def Fisher_Matrix(params,cov,model,show_results=True):
    """
    Returns the Fisher matrix for the desired parameters and the ordering. The rows and columns are in
    alphabetical order. 
    Preservation of the order in params is not gaurenteed.
    
    params: A dictionary of all parameters needed for the supplied model, including parameters that will not be 
        in the Fisher matrix. Parameter to be included in the matrix should be instances of fish_param.
    cov: The covariance for the supplied model
    model: A model which derivatives will be taken with respect to. Takes params as input. 
    """    
    params = OrderedDict(sorted(params.items()))
    if show_results==False:    
        return (add_priors(params,Fisher_Calc(derivative(model,params),cov)), [k for k,v in params.items()
                                                                               if isinstance(v,fish_param)])
    else:
        FM=(add_priors(params,Fisher_Calc(derivative(model,params),cov)), [k for k,v in params.items()
                                                                           if isinstance(v,fish_param)])
        
        for i,x in enumerate(zip(FM[1],sqrt(diag(inv(FM[0]))))):
            print(i,x[0],x[1])
        return FM
        