# create function to compute drawdown, peak etc
import pandas as pd
import scipy.stats 
import numpy as np
from scipy.stats import norm

def var_gaussian(r, level=5, modified=False):
    """
    Returns the parametric Gaussian VaR of a series or Dataframe
    
    """
    
    #compute the Z score assuming it was Gaussian
    z= norm.ppf(level/100)
    if modified:
        # modify the z score based on observed skewness and kutosis
        s = skewness(r)
        k = kurtosis(r)
        z = (z+ 
            (z**2 - 1)*s/6 + 
            (z**3 - 3*z) *(k-3)/24 -
            (2*z**3 - 5*z)*(s**2)/36
            )
    return -(r.mean() + z*r.std(ddof=0))

def drawdown(return_series: pd.Series):   # expect a pd series as input
    """ 
    Take as timeseries of asset returns  
    computes and returns a dataframe that containes:
    the wealth index
    the previos peaks
    percent drawdown
    """
    wealth_index = 1000*(1+ return_series).cumprod()
    previous_peaks = wealth_index.cummax()
    drawdowns = (wealth_index - previous_peaks)/ previous_peaks
    
    #dummy = pd.Dataframe({"W": wealth_index, "P" : previous_peaks, "D": drawdowns })
    d = pd.DataFrame({"Wealth": wealth_index, "Peaks": previous_peaks, "Drawdown" : drawdowns})
    return d

def get_ffme_returns():
    """
    Load the File
    """
    me_m = pd.read_csv("Portfolios_Formed_on_ME_monthly_EW.csv", header = 0, index_col=0, parse_dates =True, na_values=-99.99)
    rets = me_m[['Lo 10', 'Hi 10']]
    rets.columns = ['SmallCap', 'LargeCap']
    rets = rets/100
    rets.index = pd.to_datetime(rets.index, format="%Y%m").to_period('M')
    return rets

def get_hfi_returns():
    """
    Load the File
    """
    hfi = pd.read_csv("edhec-hedgefundindices.csv", header = 0, index_col=0, parse_dates =True, na_values=-99.99)
    hfi = hfi/100
    hfi.index = hfi.index.to_period('M')
    return hfi

def get_ind_returns():
    """
    Load and format the Ken French industry file
    """
    ind = pd.read_csv("ind30_m_vw_rets.csv", header = 0, index_col=0, parse_dates =True, na_values=-99.99)/100
    ind.index = pd.to_datetime(ind.index, format="%Y%m").to_period('M')
    ind.columns = ind.columns.str.strip()
    return ind


def semideviation(r):
    """
    Returns the semideviation aka negative semideviation of r
    r must be a series or a dataframe
    """
    is_negative = r < 0
    return r[is_negative].std(ddof=0)


def skewness(r):
    """
    Alternative  to scipy.stats.skewness().
    computes the skewness of the supplier series or dataframe
    returns a float or a series
    """
    demeaned_r = r - r.mean()
    # use the population standard deviation, so set ddof = 0
    sigma_r = r.std(ddof=0)
    exp = (demeaned_r**3).mean()
    return exp/sigma_r**3

def kurtosis(r):
    """
    Alternative  to scipy.stats.kurtosis().
    computes the kutosis of the supplier series or dataframe
    returns a float or a series
    """
    demeaned_r = r - r.mean()
    # use the population standard deviation, so set ddof = 0
    sigma_r = r.std(ddof=0)
    exp = (demeaned_r**4).mean()
    return exp/sigma_r**4


def is_normal(r, level=0.01):
    """
    """
    statistic, p_value = scipy.stats.jarque_bera(r)
    return p_value > level

def var_historic(r,level=5):
    """
    Return the historic Value at Risk at a specified level
    i.e returns the number such that 'level' percent of the return fall     below that number, and the (100-level) percent are above
    """
    if isinstance(r, pd.DataFrame):
        return r.aggregate(var_historic,level=level)
    elif isinstance(r,pd.Series):
        return -np.percentile(r,level)
    else:
        raise TypeError("Expected r to be series or Dataframe")
        
        
def cvar_historic(r, level=5):
    """
    Computes the CVaR of a series or dataframe
    """
    
    if isinstance(r,pd.Series):
        is_beyond = r <= -var_historic(r, level=level)
        return -r[is_beyond].mean()
    elif isinstance(r, pd.DataFrame):
        return r.aggregate(cvar_historic, level=level)
    
    else:
        raise TypeError("Expected r to be a Series or DataFrame")
        
def annualize_vol(r, periods_per_year):
    """
    
    """
    return r.std()*(periods_per_year**0.05)

def annualize_rets(r, periods_per_year):
    """
    
    """
    compounded_growth = (1+r).prod()
    n_periods = r.shape[0]    
    return compounded_growth**(periods_per_year/n_periods) - 1

def sharpe_ratio(r, riskfree_rate, periods_per_year):
    """
    
    """
    rf_per_period = (1+riskfree_rate)**(1/periods_per_year)-1
    excess_ret =  r - rf_per_period
    ann_ex_ret = annualize_rets(excess_ret, periods_per_year)
    ann_vol = annualize_vol(r, periods_per_year)
    return ann_ex_ret / ann_vol
    