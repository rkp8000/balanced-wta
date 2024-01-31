import numpy as np
from scipy import stats
import sys



def norm(x, mu, sgm):
    if sgm == 0:
        temp = np.zeros(x.shape)
        dx = np.mean(np.diff(x))
        temp[np.argmin(np.abs(x))] = 1/dx
        return temp
    
    return stats.norm.pdf(x, loc=mu, scale=sgm)


def phi(x, mu, sgm):
    if sgm == 0:
        temp = (x >= 0).astype(float)
        temp[x == 0] = .5
        return temp
    
    return stats.norm.cdf(x, loc=mu, scale=sgm)


class MFT(object):
    """Class for running MFT smln."""
    
    def __init__(self, x_lo, x_hi, dx):
        self.dx = dx
        self.x = np.arange(x_lo, x_hi, dx)

    def alph(self, r, u, v, D, mu_j, sgm_j, N=np.inf):
        """
        r \in [0, 1]^D
        u \in R^D
        v \in R_+^D
        D scalar
        mu_j \in R^{DxD}
        sgm_j \in R^{DxD}
        """
        x = self.x
        dx = self.dx
        
        mu_d = D*(mu_j @ r) + u
        sgm_d = np.sqrt(D*((sgm_j**2)@r) + v)

        phis = np.array([phi(x, mu_d_, sgm_d_) for mu_d_, sgm_d_ in zip(mu_d, sgm_d)])

        mnot_d = ~np.eye(len(r), dtype=bool) # masks for selecting all but one d

        r_next = np.nan*np.zeros(D)
        for d, (mu_d_, sgm_d_) in enumerate(zip(mu_d, sgm_d)):

            mask_d = mnot_d[d]
            r_next[d] = min(np.sum(norm(x, mu_d_, sgm_d_) * np.prod(phis[mask_d, :], axis=0))*dx, 1)

        if np.isinf(N):
            return r_next
        else:
            return np.random.multinomial(N, r_next)/N
        