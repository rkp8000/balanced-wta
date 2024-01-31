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
        """J is an DxDxNxN array"""
        assert J.ndim == 4
        
        self.J = J
        self.D = self.J.shape[0]
        self.N = self.J.shape[-1]
        
    def run(self, g, t_max, y_0=None, x_0=None, us=None, progress=0):
        """us is external input in dict form us = {t_0: u_0, t_1: u_1, ...} since we assume sparse"""
        xs = np.nan*np.zeros((t_max, self.N, self.D))
        ys = np.nan*np.zeros((t_max, self.N, self.D))
        
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
                sys.stdout.flush()
            
            # build input from components
            x = np.zeros((self.N, self.D))
            
            # recurrent
            for d_to in range(self.D):
                for d_from in range(self.D):
                    
                    x[:, d_to] += self.J[d_to, d_from]@ys[t_-1, :, d_from]
                    
            # external
            u = us[t_] if t_ in us else 0
            x += u
                
            xs[t_, :, :] = x
            ys[t_, :, :] = gsoftmax(x, g)
            
        return t, xs, ys


class SoftmaxNtwk2(object):
    """
    Version of SoftmaxNtwk in which approximately independent network weights are generated on the fly
    via scaling/shifting and randomly permuting a single connection matrix,
    enabling much larger simulations to run without having to create a huge
    dense weight tensor.
    """
    
    def __init__(self, N, MU_J, SGM_J):
        self.N = N
        
        self.D = len(MU_J)
        
        self.MU_J = MU_J
        self.SGM_J = SGM_J
        
        self.J_0 = np.random.randn(N, N)
        
        # make mean and std exactly 0 and 1
        self.J_0 -= np.mean(self.J_0)
        self.J_0 /= np.std(self.J_0)
        
        # create permutation vectors
        self.p_rows = {}
        self.p_cols = {}
        
        for d_to in range(self.D):
            for d_from in range(self.D):
                
                if (MU_J[d_to, d_from] == 0) and (SGM_J[d_to, d_from] == 0):
                    continue
                    
                self.p_rows[(d_to, d_from)] = np.random.permutation(N)
                self.p_cols[(d_to, d_from)] = np.random.permutation(N)
                
        # create realistic sample means and stds of blocks
        self.mu_j_sample = np.zeros((self.D, self.D))
        self.sgm_j_sample = np.zeros((self.D, self.D))
        
        for d_to in range(self.D):
            for d_from in range(self.D):
                
                if (MU_J[d_to, d_from] == 0) and (SGM_J[d_to, d_from] == 0):
                    continue
                    
                # generate sample mean
                self.mu_j_sample[d_to, d_from] = np.random.normal(
                    MU_J[d_to, d_from]*self.D/self.N,
                    SGM_J[d_to, d_from]*np.sqrt(self.D/self.N)*1/self.N,
                )
                
                # generate sample std
                var_j = (SGM_J[d_to, d_from]**2)*self.D/self.N
                var_j_sample = np.random.chisquare(self.N**2-1)*var_j/(self.N**2-1)
                
                self.sgm_j_sample[d_to, d_from] = np.sqrt(var_j_sample)
                
                
    def run(self, g, t_max, y_0=None, x_0=None, us=None, progress=False):
        
        xs = np.nan*np.zeros((t_max, self.N, self.D))
        ys = np.nan*np.zeros((t_max, self.N, self.D))
        
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
                sys.stdout.flush()
            
            # build input from components
            x = np.zeros((self.N, self.D))
            y_prev = ys[t_-1, :, :]
            
            # recurrent
            for d_to in range(self.D):
                for d_from in range(self.D):
                    
                    if (self.MU_J[d_to, d_from] == 0) and (self.SGM_J[d_to, d_from] == 0):
                        continue
                        
                    p_row = self.p_rows[(d_to, d_from)]
                    p_col = self.p_cols[(d_to, d_from)]
                    
                    mu_j_sample = self.mu_j_sample[d_to, d_from]
                    sgm_j_sample = self.sgm_j_sample[d_to, d_from]
                    
                    temp_1 = y_prev[p_col, d_from]
                    
                    temp_2 = sgm_j_sample*(self.J_0@temp_1) + mu_j_sample*temp_1
                    
                    x[:, d_to] += temp_2[p_row]
                    
            # external
            u = us[t_] if t_ in us else 0
            x += u
                
            xs[t_, :, :] = x
            ys[t_, :, :] = gsoftmax(x, g)
        
        return t, xs, ys
