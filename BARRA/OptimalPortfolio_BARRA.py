import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import math
import matplotlib.pyplot as plt
from cvxopt import matrix, solvers


# 离群值处理
def outlier_func(s):
    q01 = s.quantile(q=0.01)
    q99 = s.quantile(q=0.99)

    s = np.minimum(s,q99)
    s = np.maximum(s,q01)
    return s

# 回归函数参数
def regre_params(x, formula):
    return smf.ols(formula, data=x).fit().params

# 回归函数残差
def regre_resid(x, formula):
    return smf.ols(formula, data=x).fit().resid

# 获取对应时间节点的股票dataframe
def stock_df(proj_dataframe, t=120):
    # get the 120 months dates
    date_list = proj_dataframe['DATE'].unique()[t-120:t].tolist()
    # get the list of stock in the target date
    date = date_list[-1]
    target_stock_list = proj_dataframe.loc[proj_dataframe['DATE'] == date]['PERMNO'].to_list()
    # construct the regression dataframe according to date and stock list
    reg_project = proj_dataframe.loc[proj_dataframe['DATE'].isin(date_list)]
    # eliminate stocks that have less than 100 month data
    reg_project = reg_project.loc[reg_project['PERMNO'].isin(target_stock_list)].groupby('PERMNO').filter(lambda x : len(x) > 100)
    reg_project = reg_project.set_index('PERMNO')    
    
    return reg_project
 
# 计算样本周期内收益率矩阵和协方差矩阵
def char_variance_matrix(reg_project_dataframe):
    '''
    reg_project_dataframe: 存入的前一百二十个月股票特征数据
    '''
    reg_params = reg_project_dataframe.groupby('DATE').apply(regre_params, 'exret ~ mkt_beta + logme + logbeme + r_2_12 + gp + invest_asset - 1')
    params_var = reg_params.cov(ddof=6)
    
    current_date = reg_project_dataframe.DATE.tolist()[-1]
    reg_project_dataframe = reg_project_dataframe.loc[reg_project_dataframe['DATE'] == current_date][['mkt_beta', 'logme', 'logbeme', 'r_2_12', 'gp', 'invest_asset']]

    reg_var = np.dot(reg_project_dataframe, np.dot(params_var, reg_project_dataframe.T))
    
    return reg_var


def residual_matrix(reg_project_dataframe):
    
    reg_resid = reg_project_dataframe.groupby('DATE').apply(regre_resid, 'exret ~ mkt_beta + logme + logbeme + r_2_12 + gp + invest_asset - 1')
    reg_resid_df = reg_resid.to_frame(name='resid').reset_index()
    reg_resid_df = reg_resid_df.set_index(['DATE','PERMNO'])['resid'].unstack().rename_axis(columns=None).reset_index()
    # 残差的方差
    resid_var = pd.Series(reg_resid_df.set_index('DATE').var())
    
    resid_var_df = pd.DataFrame(0, index=resid_var.index, columns=resid_var.index, dtype=resid_var.dtype)
    # 对角线填入
    np.fill_diagonal(resid_var_df.values, resid_var)
    
    return resid_var_df

# 预期股票收益率协方差矩阵为 因子暴露的转置与因子方差点乘因子暴露 加上残差方差矩阵
def covariance_matrix(reg_project_dataframe):
    
    a = char_variance_matrix(reg_project_dataframe)
    b = residual_matrix(reg_project_dataframe)
    
    return a+b

# 预期收益率向量为股票因子暴露与相应因子预测收益率相乘之和。
def return_col(reg_project_dataframe):
    
    reg_params = reg_project_dataframe.groupby('DATE').apply(regre_params, 'exret ~ mkt_beta + logme + logbeme + r_2_12 + gp + invest_asset - 1')
    estimate_params = reg_params.mean()
    
    current_date = reg_project_dataframe.DATE.tolist()[-1]
    reg_project_dataframe = reg_project_dataframe.loc[reg_project_dataframe['DATE'] == current_date][['mkt_beta', 'logme', 'logbeme', 'r_2_12', 'gp', 'invest_asset']]
    ret_col = reg_project_dataframe.dot(estimate_params)
    
    return ret_col

# 求解最小方差组合
def minimal_variance_portfolio(cov_matrix, date):
    
    e = np.ones(len(cov_matrix))
    e = pd.DataFrame(e,index=cov_matrix.index)

    cov_matrix_inv = pd.DataFrame(np.linalg.inv(cov_matrix.values), cov_matrix.columns, cov_matrix.index)

    weights = np.dot(cov_matrix_inv,e)/np.dot(np.transpose(e), np.dot(cov_matrix_inv, e))
    weights = pd.DataFrame(weights, index=cov_matrix.index, columns=[date], dtype='double')
    
    return weights


# 求解BARRA long-short组合
# BARRA Mean-Var Portfolio的计算
def barra_mean_variance_portfolio(cov_matrix, return_col, reg_project_opt, date):
    '''
    使用cvxopt求解二次规划, 计算最优权重
    '''
    P = (3/2)*matrix(np.array(cov_matrix))
    q = -matrix(np.array(return_col.T))
    
    current_date = int(reg_project_opt['DATE'][-1:].values)
    beta = reg_project_opt[(reg_project_opt.index.isin(return_col.index)) & (reg_project_opt['DATE'] == current_date)]['mkt_beta']
    log_ME = reg_project_opt[(reg_project_opt.index.isin(return_col.index)) & (reg_project_opt['DATE'] == current_date)]['logme']

    A = matrix(np.array([[1 for _ in range(len(cov_matrix))] , beta, log_ME]))
    b = matrix([0,0,0], tc='d')
    G = matrix(np.r_[np.identity(len(cov_matrix)), -np.identity(len(cov_matrix))])
    h = matrix(np.array([0.01 for _ in range(np.array(G).shape[0])]))
    
    sol = solvers.qp(P, q, G, h, A, b)
    weights = pd.DataFrame(np.array(sol['x']), index=return_col.index, columns=[date], dtype='double')
    
    return weights

# 根据权重计算收益率
def cal_daily_return(weights_df, proj_data, date_list, rf):
    ret_data = pd.DataFrame(columns=['monthly_exret'])
    
    for day in date_list:
        day_weight = weights_df.loc[day].dropna()
        day_weight.index = day_weight.index.astype('int')
        
        day_ret = final_proj[(final_proj['DATE'] == day) & (final_proj['PERMNO'].isin(day_weight.index.astype('int').tolist()))][['PERMNO', 'exret']]
        day_ret = day_ret.set_index('PERMNO')
        
        day_weight = day_weight[day_ret.index.tolist()]
        ret = day_ret.T.dot(day_weight)
        ret.index = [day]
        
        df = pd.DataFrame(ret, columns=['monthly_exret'])
        ret_data = pd.concat([ret_data, df])
    
    ret_data = ret_data.join(rf)
    ret_data['monthly_return'] = ret_data['monthly_exret'] + ret_data['RF']
        
    return ret_data


# 评价指标
def ret_performance(ret_data):
    '''
    annualized_ret: 年化收益, %
    annualized_exret: 年华超额收益, %
    annualized_std: 年华波动率, %
    sharpe_ratio: 夏普比率
    maximum_drawdown: 最大回撤, %
    recover_period: 最大 回撤回复周期, month
    '''
    
    ret_data.index = ret_data.index.astype('str')
    
    annualized_ret =  ret_data['monthly_return'].mean() * 12
    annualized_exret = ret_data['monthly_exret'].mean() * 12
    annualized_std = ret_data['monthly_exret'].std() * math.sqrt(12)
    sharpe_ratio = annualized_exret/annualized_std
    
    ret_data['cumret'] = (ret_data['monthly_return'] + 1).cumprod()
    ret_data['drawdown'] = (ret_data['cumret'].cummax() - ret_data['cumret']) / ret_data['cumret'].cummax()
    maximum_drawdown = ret_data['drawdown'].max()
    recover_period = ret_data['cumret'].cummax().value_counts().max()
    
    evaluation = {'annualized_ret' : annualized_ret, 'annualized_exret' : annualized_exret, 'annualized_std': annualized_std, 'sharpe_ratio': sharpe_ratio, 'maximum_drawdown': maximum_drawdown, 'recover_period' : recover_period}
    result = {'evaluation': evaluation, 'ret_data' : ret_data}

    return result


# 根据月收益率数据绘制收益率曲线
def plot_ret(ret_data):
    
    ret_data.index = ret_data.index.astype('str')
    
    ret_data['cumret'] = (ret_data['monthly_return'] + 1).cumprod()
    ret_data['cumret'].plot(label='month return curve',figsize=(20,10), grid=True)
    
    plt.legend()
    plt.show()

### 处理数据
# 读取数据
project = pd.read_csv(r'python\QA_FinalProject\finalproj.csv')
# project = pd.read_csv(r'./quant_project/finalproj.csv')
project = project.dropna()
riskfactor = pd.read_csv(r'python\QA_FinalProject\riskfactor.csv')
# riskfactor = pd.read_csv(r'./quant_project/riskfactor.csv')
# 公司特征变量名称
firm_char = ['logme', 'logbeme', 'r_2_12', 'gp', 'invest_asset']

# 不改变原数据
final_proj = project.copy()
# Data Preprocessing, handling outlier and standardized
final_proj = final_proj.groupby('DATE').apply(lambda x : x.apply(lambda y : outlier_func(y) if y.name in firm_char else y))
final_proj = final_proj.groupby('DATE').apply(lambda x : x.apply(lambda y: ((y - y.mean()) / y.std())if y.name in firm_char else y))

### 执行函数
weights_df = pd.DataFrame()
for t in range(120, len(riskfactor)):
    reg_project = stock_df(final_proj, t=t)
    
    covariance_matirx = covariance_matrix(reg_project)
    ret_col = return_col(reg_project)
    weights = barra_mean_variance_portfolio(covariance_matirx, ret_col, reg_project, date=riskfactor.date.tolist()[t])

    weights_df = pd.concat([weights_df, weights.T])

# weights_df.to_csv('./barra_optimal_weight.csv', index=True)
print(weights_df)
