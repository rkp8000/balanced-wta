import numpy as np
from scipy import stats
import sys


def norm(x, mu, sgm):
    if sgm == 0:
        temp = np.zeros(X.shape)
        temp[np.argmin(np.abs(X))] = 1/DX
        return temp
    
    return stats.norm.pdf(x, loc=mu, scale=sgm)


def phi(x, mu, sgm):
    if sgm == 0:
        temp = (X >= 0).astype(float)
        temp[X == 0] = .5
        return temp
    
    return stats.norm.cdf(x, loc=mu, scale=sgm)


def alph(th, u, v, R, mu_j, sgm_j, N=np.inf):
    """
    th \in [0, 1]^R
    u \in R^R
    v \in R_+^r
    R scalar
    mus \in R^{RxR}
    gams \in R^{RxR}
    """
    mu_r = R*(mu_j @ th) + u
    sgm_r = np.sqrt(R*((sgm_j**2)@th) + v)
    
    phis = np.array([phi(X, mu_r_, sgm_r_) for mu_r_, sgm_r_ in zip(mu_r, sgm_r)])
    
    mnot_r = ~np.eye(len(th), dtype=bool) # masks for selecting all but one r
    
    th_next = np.nan*np.zeros(R)
    for r, (mu_r_, sgm_r_) in enumerate(zip(mu_r, sgm_r)):
        
        mask_r = mnot_r[r]
        th_next[r] = min(np.sum(norm(X, mu_r_, sgm_r_) * np.prod(phis[mask_r, :], axis=0))*DX, 1)
        
    if np.isinf(N):
        return th_next
    else:
        return np.random.multinomial(N, th_next)/N
    

def make_alph(xr):
    """
    xr: x vector for calculating integral
    """
    dx = np.mean(np.diff(xr))
    
    