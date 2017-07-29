
from copy import deepcopy
from numpy import dot, array, zeros, diag, sqrt, arange, shape
from scipy.linalg import cho_factor, cho_solve, inv
from collections import OrderedDict


#When using from Fisher_CMB import * only the follwing line will be imported
__all__ = ['fish_param','Fisher_Matrix']


def Fisher_Matrix(params,cov,model,show_results=True):
    """
    Returns the Fisher matrix for the desired parameters and the ordering. The rows and columns are in
    alphabetical order. 
    Preservation of the order in params is not gaurenteed.
    
    params: A dictionary of all parameters needed for the supplied model, including parameters that will not be 
        in the Fisher matrix but needed for each call of the model. Parameters to be included in the matrix should be instances           of fish_param.
        
        Example: params={'H0':None,'cosmomc_theta': fish_param(1.0438e-2,1.3e-5),
                           'ombh2':fish_param(0.0223,0.001),
                           'ommh2':fish_param(0.13,0.007),
                           'tau':fish_param(0.083,0.02, 0.02),
                           'clamp':fish_param(1.93e-09,4.5e-11),
                           'ns':fish_param(0.9623,0.032),
                           'nnu':fish_param(3.046,0.03),
                           'lmax':6000,'lens_potential_accuracy':2.0}
                 The last two parameters are needed by CAMB but not included in the fisher matrix.
                           
    cov: The covariance for the supplied model
    model: A model which derivatives will be taken with respect to. Takes params as input. 
    show_results: print out the ordering of the parameters in the Fisher matrix and the constraints
    
    returns: A tuple whose first entry is the fisher matrix and the second entry is the order of the parameters
    """
    # Turn params into OrderedDict object to keep everything in order
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
        
class fish_param(object):
    '''
    Parameters that are to be included in the fisher matrix should be instances of fish_param in the params dictionary
    that is given to Fisher_Matrix; fish_params have the attributes, base, step and prior (optional).
    
    Base: value which derivative will be taken around.
    Step: step size for derivative.
    Prior: prior contraint
    '''
    def __init__(self,base,step,prior=None):
        self.base = base
        self.step = step
        self.prior = prior


def derivative(model,params):
    '''
    Calculate the partial derivatives of a model with respect to the parameters to be included in the 
    Fisher matrix using finite differences. 
    '''
    dydx=[]
    base={l:m.base for l,m in params.items() if  isinstance(m,fish_param)}
    for k,v in params.items():
        if isinstance(v,fish_param): #take derivatives with respect to only fish_params
            tmp = deepcopy(params) #use deepcopy so tmp does not point to params
            tmp.update(base) #this update replaces all fish_params with base value and can now be fed to model
            tmp[k]+=v.step/2
            modelu = model(**tmp)
            tmp[k]-=v.step
            modeld = model(**tmp)
            dydx.append((modelu-modeld)/v.step) 
    return array(dydx)


def Fisher_Calc(model_der,cov):
    """
    Calculate the Fisher Matrix for a given model and covariance. Use the cholesky decomposition for
    faster calculation than inv().
    """
    cho_cov = cho_factor(cov)
    return dot(model_der,cho_solve(cho_cov,model_der.T))


def add_priors(params,Fisher_matrix):
    '''
    Adds the priors to the completed fisher matrix
    '''
    priors = [ m.prior for l,m in params.items() if isinstance(m,fish_param)]
    for i,p in enumerate(priors):
        if p is not None:
            Fisher_matrix[i,i]+=p**(-2)
    return Fisher_matrix
            
            
