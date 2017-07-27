# CMB-Forcasting
The main file is FIsher_CMB.py.
This contains all of the general fisher matrix calculation stuff. To use it you want to run "from Fisher_CMB import *"
You can then get the fisher matrix by by running FIsher_Matrix(params,cov, model).
params is a dictionary that contains all of the parameters needed for each call to model. The parameters that you want 
in the fisher matrix need to be instances of fish_param(value,step,prior). Cov is just the covariance that corresponds to the 
output of model and model is a function that when given params (model(**params)) returns an array. 

In the Models directory ( I need to organize this better) is the code for SPT3G, and Planck models. the CMB directory 
contains the code for making calls to CAMB. COVS has the code I use to create a covariance. The main function is
create_basic_cov. It has several inputs for noise, sky coverage, etc.. It calculates the various matrices for TTTT,TTTE,TTEE,EEE, etc.
and saves each one as a dictionary. I went this route because it is a smaller save file than just saving the entire matrix and takes less time,
it is also easier to access, say, the EEEE matrix to check it. There is another function that takes this dictionary and converts it to the full covariance
called make_full_cov(). The first arg is a list of the spectra the model outputs and also needs to be in the same order. The second arg is 
just the dictionary of covs, TTTT and so on.

That should be about it. There is a notebook where you can see how I use things, but it is a bit sloppy, I have not had the time to comment on everything.
