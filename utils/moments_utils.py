'''
Created on May 6, 2025

@author: bernardo
'''

import numpy as np
import math

def normal_moments(mu,sigma,K):
    """
    An auxiliary function that returns the first K+1 non-central moments of the normal distribution. 
    Parameters
    ----------
    mu : mean of normal dist
    sigma : std of normal dist
    K: maximum moment to return
    Returns
    -------
    moments_normal : array of shape K+1, with moments 0,1,...,K
    """
    moments_normal = np.empty((K+1,))
    moments_normal[0] = 1
    moments_normal[1] = mu
    sigma_sq = np.square(sigma)
    if K>=2:
        for k in range(2, K+1):
            moments_normal[k] = moments_normal[1]*moments_normal[k-1]+(k-1)*sigma_sq*moments_normal[k-2]
    return moments_normal

def exponential_moments(scale,K):
    """
    An auxiliary function that returns the first K+1 non-central moments of the exponential distribution. 
    Parameters
    ----------
    scale : scale of exponential dist
    K: maximum moment to return
    Returns
    -------
    moments_exp : array of shape K+1, with moments 0,1,...,K
    """
    
    moments_exp = np.empty((K + 1,))
    moments_exp[0] = 1
    
    for k in range(1, K + 1):
        moments_exp[k] = math.factorial(k) * np.power(scale, k)
    
    return moments_exp