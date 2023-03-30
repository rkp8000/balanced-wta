import numpy as np
from scipy import special
import sys


# softmax function that accepts infinite gain
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
    
    
# simplest network implementation
class SoftmaxNtwk(object):
    
    def __init__(self, J):
        """J is an RxRxNxN array"""
        assert J.ndim == 4
        
        self.J = J
        self.R = self.J.shape[0]
        self.N = self.J.shape[-1]
        
    def run(self, g, t_max, y_0=None, x_0=None, us=None, progress=0):
        """us is external input in dict form us = {t_0: u_0, t_1: u_1, ...} since we assume sparse"""
        xs = np.nan*np.zeros((t_max, self.N, self.R))
        ys = np.nan*np.zeros((t_max, self.N, self.R))
        
        if y_0 is not None:
            ys[0, :, :] = y_0
            
        elif x_0 is not None:
            xs[0, :, :] = x_0
            ys[0, :, :] = gsoftmax(x_0, g)
            
        if us is None:
            us = {}
            
        t = np.arange(t_max)
        
        for t_ in t[1:]:
            
            if progress > 0 and (t_%progress) == 0:
                sys.stdout.write('.')
            
            # build input from components
            x = np.zeros((self.N, self.R))
            
            # recurrent
            for r_to in range(self.R):
                for r_from in range(self.R):
                    
                    x[:, r_to] += self.J[r_to, r_from]@ys[t_-1, :, r_from]
                    
            # external
            u = us[t_] if t_ in us else 0
            x += u
                
            xs[t_, :, :] = x
            ys[t_, :, :] = gsoftmax(x, g)
            
        return t, xs, ys


class SoftmaxNtwk2(object):
    """Version of SoftmaxNtwk in which quenched ntwk connectivity is generated on the fly
    via scaling and randomly permuting single connection matrix, enabling much larger
    simulations to run quickly."""
    
    def __init__(self, D, N):
        self.D = D
        self.N = N
        
        self.j_0 = np.random.randn(N, N)
        
        # create permutation vectors
        self.p_rows = np.zeros((D, D, N), dtype=np.int32)
        self.p_cols = np.zeros((D, D, N), dtype=np.int32)
        for d in range(D):
            for d_prime in range(D):
                self.p_rows[d, d_prime, :] = np.random.permutation(N)
                self.p_cols[d, d_prime, :] = np.random.permutation(N)
                
    def run(self, t_max, g, mu_0, mu_1, gam, x_0=None, y_0=None, progress=False):
        
        xs = np.nan*np.zeros((t_max, self.N, self.D))
        ys = np.nan*np.zeros((t_max, self.N, self.D))
        
        # initial conditions
        xs[0, :, :] = np.random.randn(self.N, self.D) if x_0 is None else x_0
        ys[0, :, :] = softmax(x[0, :, :], g) if y_0 is None else y_0
        
        # main loop
        t = np.arange(t_max, dtype=np.int16)
        
        for t_ in t[1:]:
            if progress:
                sys.stdout.write('>')
            
            x_next = np.zeros((self.N, self.D))
            y_prev = ys[t_-1, :, :]
            
            # loop over targ labels
            for d in range(self.D):
                for d_prime in range(self.D):
                    
                    p_row = self.p_rows[d, d_prime]
                    p_col = self.p_cols[d, d_prime]
                    
                    temp = (self.j_0@y_prev[p_col, d_prime])[p_row]
                    
                    if d == d_prime:
                        x_next[:, d] += (np.sqrt(self.D/self.N)*temp) + mu_0*(self.D/self.N)*np.sum(y_prev[:, d_prime])
                    else:
                        x_next[:, d] += gam*np.sqrt(self.D/self.N)*temp + mu_1*self.D/self.N*np.sum(y_prev[:, d_prime])

            ys[t_, :, :] = softmax(x_next, g)
        
        return t, xs, ys
    