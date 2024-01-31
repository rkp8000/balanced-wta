import numpy as np

def get_abins(x, nbin):
    """Divide data into bins of different widths each containing approx 1/nbin data points."""
    assert len(x) >= nbin
    
    bins = np.quantile(x, np.linspace(0, 1, nbin+1))
    binc = .5*(bins[:-1] + bins[1:])
    
    return bins, binc