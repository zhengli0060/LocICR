import math
import numpy as np
from math import log, sqrt
from scipy.stats import norm
from sklearn.linear_model import Lasso

def CI_Test_lasso(X, Y, S, D, alpha):
    # Input arguments:
    # X, Y: the variables to check the conditional independence of
    # S: conditioning set
    # D: Data matrix (number_of_Samples * n)
    # alpha: the parameter of Fischer's Z transform

    # Output arguments:
    # CI: the conditional independence relation between X, Y,
    #    true if independent, false if dependent
    # p: p-value (The null hypothesis is for independence)
    X = int(X)
    Y = int(Y)
    if isinstance(S, np.ndarray):
        S = S.tolist()
    n = D.shape[0]
    #------------lasso--------------
    A = D[:, X]
    B = D[:, Y]
    con_set_data = D[:, S]
    con_set_mean = np.mean(con_set_data, axis=0)
    con_set_centered = con_set_data - con_set_mean
    con_set_std = np.std(con_set_centered, axis=0)
    con_set_standardized = con_set_centered / con_set_std
    #------------X   S ------------
    lasso_model = Lasso(alpha=0.05)
    lasso_model.fit(con_set_standardized, A)
    feature_importance = lasso_model.coef_
    selected_features = np.array(range(len(feature_importance)))[feature_importance != 0]
    con1 = []
    for i in selected_features:
        con1.append(S[i])
    # ------------Y   S ------------
    lasso_model = Lasso(alpha=0.05)
    lasso_model.fit(con_set_centered, B)
    feature_importance = lasso_model.coef_
    selected_features = np.array(range(len(feature_importance)))[feature_importance != 0]
    con2 = []
    for i in selected_features:
        con2.append(S[i])

    con_set = list(set(con1 + con2))


    DD = D[:, [X, Y, *con_set]]
    corr_matrix = np.corrcoef(DD, rowvar=False)
    try:
        inv = np.linalg.inv(corr_matrix)
    except np.linalg.LinAlgError:
        raise ValueError('Data correlation matrix is singular. Cannot run fisherz test. Please check your data.')
    r = -inv[0, 1] / sqrt(abs(inv[0, 0] * inv[1, 1]))
    if abs(r) >= 1: r = (1. - np.finfo(float).eps) * np.sign(r)
    Z = 0.5 * log((1 + r) / (1 - r))
    X = sqrt(n - len(con_set) - 3) * abs(Z)
    p_value = 2 * (1 - norm.cdf(abs(X)))
    CI = p_value > alpha

    return CI, p_value

def MB_TC(D, A, alpha):
    # Total Conditioning, see Pellet and Elisseeff
    n = D.shape[1]
    do_n = n / 10
    ceil_result = math.floor(do_n)
    if ceil_result > 0:
        alpha = alpha/(ceil_result*10)
    # print(f"MMB_TC alpha :{alpha}")
    nVars = D.shape[1]
    nTests = 0
    A_MMB = []
    X = A
    tmp = list(range(0, nVars))
    tmp.remove(X)
    for Y in tmp:
            S = list(range(0, nVars))
            S.remove(X)
            S.remove(Y)
            CI, p = CI_Test_lasso(X, Y, S, D, alpha)
            nTests += 1
            if not CI:#not independent
                A_MMB.append(Y)

    return A_MMB,nTests

