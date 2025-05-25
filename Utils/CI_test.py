import numpy as np
from scipy.stats import norm
from math import log, sqrt
from typing import Union, Set, Tuple, Dict, Any, List
from numpy.linalg import LinAlgError
# def CI_Test(X: int, Y: int, S: List[int], D: np.ndarray, alpha: float) -> Tuple[bool, float]:
#     # Input arguments:
#     # X, Y: the variables to check the conditional independence of
#     # S: conditioning set
#     # D: Data matrix (number_of_Samples * n)
#     # alpha: the parameter of Fischer's Z transform

#     # Output arguments:
#     # CI: the conditional independence relation between X, Y,
#     #    true if independent, false if dependent
#     # p: p-value (The null hypothesis is for independence)
#     X = int(X)
#     Y = int(Y)
#     
#     if isinstance(S, np.ndarray):
#         S = S.tolist()
#     # if Y in S:
#     #     return True, 0
#     n = D.shape[0]
#     DD = D[:, [X, Y, *S]]
#     corr_matrix = np.corrcoef(DD, rowvar=False)
#     try:
#         inv = np.linalg.inv(corr_matrix)
#     except np.linalg.LinAlgError:
#         raise ValueError('Data correlation matrix is singular. Cannot run fisherz test. Please check your data.')
#     r = -inv[0, 1] / sqrt(abs(inv[0, 0] * inv[1, 1]))
#     if abs(r) >= 1: r = (1. - np.finfo(float).eps) * np.sign(r)

#     Z = 0.5 * log((1 + r) / (1 - r))
#     X = sqrt(n - len(S) - 3) * abs(Z)
#     p_value = 2 * (1 - norm.cdf(abs(X)))
#     CI = p_value > alpha

#     return CI, p_value


def CI_Test(X: int, Y: int, S: List[int], suffStat: dict['C': np.ndarray, 'n': int], alpha: float) -> Tuple[bool, float]:
    """
    Perform Gaussian Conditional Independence Test.
    
    Parameters:
    X, Y: int - Variables to test for conditional independence.
    S: list - Conditioning set.
    suffStat: dict - Sufficient statistics containing:
        'C': Correlation matrix.
        'n': Sample size.
    
    Returns:
    float - p-value of the test.
    """
    z = zStat(X, Y, S, suffStat['C'], suffStat['n'])
    p_value = 2 * norm.cdf(-abs(z))
    CI = p_value > alpha
    return CI, p_value

def zStat(X: int, Y: int, S: list, C: np.ndarray, n: int) -> float:
    """
    Calculate Fisher's z-transform statistic of partial correlation.
    
    Parameters:
    X, Y: int - Variables to test for conditional independence.
    S: list - Conditioning set.
    C: np.ndarray - Correlation matrix.
    n: int - Sample size.
    
    Returns:
    float - z-statistic.
    """
    r = pcorOrder(X, Y, S, C)
    if r is None:
        return 0
    return np.sqrt(n - len(S) - 3) * 0.5 * np.log((1 + r) / (1 - r))

def pcorOrder(i: int, j: int, k: list, C: np.ndarray, cut_at: float = 0.9999999) -> float:
    """
    Compute partial correlation.
    
    Parameters:
    i, j: int - Variables to compute partial correlation.
    k: list - Conditioning set.
    C: np.ndarray - Correlation matrix.
    
    Returns:
    float - Partial correlation coefficient.
    """
    if len(k) == 0:
        r = C[i, j]
    elif len(k) == 1:
        r = (C[i, j] - C[i, k[0]] * C[j, k[0]]) / np.sqrt((1 - C[j, k[0]]**2) * (1 - C[i, k[0]]**2))
    else:
        try:
            sub_matrix = C[np.ix_([i, j] + k, [i, j] + k)]
            PM = np.linalg.pinv(sub_matrix)
            r = -PM[0, 1] / np.sqrt(PM[0, 0] * PM[1, 1])
        except LinAlgError:
            return None
    if np.isnan(r):
        return 0
    return min(cut_at, max(-cut_at, r))