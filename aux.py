import numpy as np
from scipy import special, stats
import string


class Generic(object):
    """Class for generic object."""
    
    def __init__(self, **kwargs):
        
        for k, v in kwargs.items():
            self.__dict__[k] = v

            
def gsoftmax(z, g):
    """Return softmax(gz) row-wise, where g -> inf is handled nicely."""
    if np.any(np.isinf(z)):
        print('Only g can be inf.')
        raise NotImplementedError
        
    z_ = z.astype(float)
    
    if np.isinf(g) and (g > 0):
        z_[z_ != z_.max(axis=1)[:, None]] = -np.inf
        return special.softmax(z_, axis=1)
    
    elif np.isinf(g) and (g < 0):
        z_[z_ != z_.min(axis=1)[:, None]] = np.inf
        return special.softmax(-z_, axis=1)
    
    else:
        return special.softmax(g*z_, axis=1)
    

def get_f_inv(x, y):
    """
    Given evenly spaced x and monotonic function values y, return a function that takes in y values
    and returns their corresponding x values.
    """
    ry = np.linspace(y.min(), y.max(), 10*len(x))
    x_f_inv = np.interp(ry, y, x)
    
    def f_inv(y_):
        iy = np.argmin(np.abs(y_ - ry))
        return x_f_inv[iy]
    
    return f_inv


def rand_string(n):
    return ''.join(np.random.choice(list(string.ascii_letters + string.digits), n))


def get_c_mean_p(a, p):
    """
    Circular mean from prob distr.
    
    a: angle vector of form np.arange(0, 2*np.pi, 2*np.pi/n)
    p: corresponding probabilities
    """
    
    return np.arctan2(p@np.sin(a), p@np.cos(a))


def get_c_spd(c_mean, t_start):
    """Estimate speed of uniform circular motion.
    c_mean in radians
    """
    t = np.arange(len(c_mean))
    
    c_mean_unw = np.unwrap(c_mean)
    
    slp = stats.linregress(t[t_start <= t], c_mean_unw[t_start <= t])[0]
    
    return slp
