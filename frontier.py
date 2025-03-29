import pandas as pd
import numpy as np
import scipy.optimize as sco
import matplotlib.pyplot as plt


def portfolio_annualised_performance(weights, mean_returns, cov_matrix) -> tuple:
    '''
        weights - веса для каждого актива list длина 30
        mean_returns - доходности активов
        cov_matrix - ковариационная матрица
        '''
    returns = np.sum(mean_returns*weights ) *252 # days
    std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
    return std, returns

# sharpe -----------------------------------------------------------------------
def neg_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate):
    p_var, p_ret = portfolio_annualised_performance(weights, mean_returns, cov_matrix)
    return -(p_ret - risk_free_rate) / p_var

def max_sharpe_ratio(mean_returns, cov_matrix, risk_free_rate, constarin_set = (0,1)):
    '''
    Minimize the negative Sharpe Ratio (SR)

    constarin_set - в каком диапазоне могут быть веса
    '''
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix, risk_free_rate)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1}) # сумма весов равна 1
    bounds = tuple(constarin_set for asset in range(num_assets))
    result = sco.minimize(neg_sharpe_ratio, num_assets*[1./num_assets,], args=args, # num_assets*[1./num_assets,] - изначальные веса для портфелей
                        method='SLSQP', bounds=bounds, constraints=constraints)
    return result

# volatility -----------------------------------------------------------------------
def portfolio_volatility(weights, mean_returns, cov_matrix):
    '''
    return std of portfolio
    '''
    return portfolio_annualised_performance(weights, mean_returns, cov_matrix)[0]

def min_variance(mean_returns, cov_matrix, constarin_set = (0,1)):
    '''
    Minimize variance of portfolio
    '''
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple(constarin_set for asset in range(num_assets))

    result = sco.minimize(portfolio_volatility, num_assets*[1./num_assets,], args=args,
                        method='SLSQP', bounds=bounds, constraints=constraints)

    return result

def efficient_return(mean_returns, cov_matrix, target, constarin_set = (0,1)):
    '''
    target - таргет по доходности (желаемая доходность)

    Для каждой доходности хотим найти портфель с минимальной волатильностью
    '''
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix)

    def portfolio_return(weights):
        return portfolio_annualised_performance(weights, mean_returns, cov_matrix)[1]

    constraints = ({'type': 'eq', 'fun': lambda x: portfolio_return(x) - target}, # return of portf (m.T@w) = target m - asset returns, w - portf weights
                   {'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple(constarin_set for asset in range(num_assets))
    result = sco.minimize(portfolio_volatility, num_assets*[1./num_assets,], args=args, method='SLSQP', bounds=bounds, constraints=constraints)
    return result


def calculated_results(mean_returns, cov_matrix, risk_free_rate, pivot_data: pd.DataFrame, constarin_set = (0,1)):
    '''
    return max SR, min Volatility, efficient frontier
    '''
    # Max SR
    maxSR_result = max_sharpe_ratio(mean_returns, cov_matrix, risk_free_rate, constarin_set)
    maxSR_std, maxSR_returns = portfolio_annualised_performance(maxSR_result['x'], mean_returns, cov_matrix) # пихаем веса из макс Шарпа
    # maxSR_allocation = pd.DataFrame(maxSR_result['x'], index=pivot_data.columns, columns=['allocation'])
    # maxSR_allocation['allocation'] = [round(i, 2) for i in maxSR_allocation['allocation']]

    # min Volatility
    minVol_result = min_variance(mean_returns, cov_matrix, constarin_set)
    minVol_std, minVol_returns = portfolio_annualised_performance(minVol_result['x'], mean_returns, cov_matrix) # пихаем веса из макс Шарпа
    # minVol_allocation = pd.DataFrame(minVol_result['x'], index=pivot_data.columns, columns=['allocation'])
    # minVol_allocation['allocation'] = [round(i, 2) for i in minVol_allocation['allocation']]


    #frontier
    target_returns = np.linspace(minVol_returns, maxSR_returns, 100)

    efficient_list =[]
    for target in target_returns:
        efficient_list.append(efficient_return(mean_returns, cov_matrix, target, constarin_set)['fun'])

    return maxSR_std, maxSR_returns, minVol_std, minVol_returns, efficient_list, target_returns
    # return maxSR_std, maxSR_returns, maxSR_allocation, minVol_std, minVol_returns, minVol_allocation, efficient_list, target_returns
    


# ----------------------------------------------------------


def random_portfolios(num_portfolios: int, mean_returns: pd.Series, cov_matrix: pd.DataFrame, risk_free_rate: float):
    results = np.zeros((3,num_portfolios))
    weights_record = []
    for i in range(num_portfolios):
        weights = np.random.dirichlet(np.ones(30))
        weights_record.append(weights)
        portfolio_std_dev, portfolio_return = portfolio_annualised_performance(weights, mean_returns, cov_matrix)
        results[0,i] = portfolio_std_dev
        results[1,i] = portfolio_return
        results[2,i] = (portfolio_return - risk_free_rate) / portfolio_std_dev # sharpe ratio
    return results, weights_record