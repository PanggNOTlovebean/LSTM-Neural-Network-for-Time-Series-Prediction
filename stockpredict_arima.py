import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import acf,pacf,plot_acf,plot_pacf
from statsmodels.tsa.arima_model import ARMA
import datetime


def main():
    # 读取股票数据
    data=pd.read_csv('data/shangzhengindex.csv')
    data=data.set_index('Date')
    data=data.drop(['code','pctChg','Volume'],axis=1)
    x=data.index.values
    y=data['Close'].values
        x= [datetime.datetime.strptime(d, '%Y-%m-%d') for d in x] 
    fig=plt.figure()
    # plt.plot(x,y)
    # plt.show()
    # 一阶差分 ADF校验
    y=data['Close'].diff(1).dropna()
    # print(y)
    t=sm.tsa.stattools.adfuller(y)
    output=pd.DataFrame(index=['Test Statistic Value', "p-value", "Lags Used", "Number of Observations Used","Critical Value(1%)","Critical Value(5%)","Critical Value(10%)"],columns=['value'])
    output['value']['Test Statistic Value'] = t[0]
    output['value']['p-value'] = t[1]
    output['value']['Lags Used'] = t[2]
    output['value']['Number of Observations Used'] = t[3]
    output['value']['Critical Value(1%)'] = t[4]['1%']
    output['value']['Critical Value(5%)'] = t[4]['5%']
    output['value']['Critical Value(10%)'] = t[4]['10%']
    # print(output)
    # ACF与PACF
    # plot_acf(y)
    # plot_pacf(y)
    # plt.show()
    r,rac,Q = sm.tsa.acf(y, qstat=True)
    prac = pacf(y,method='ywmle')
    table_data = np.c_[range(1,len(r)), r[1:],rac,prac[1:len(rac)+1],Q]
    table = pd.DataFrame(table_data, columns=['lag', "AC","Q", "PAC", "Prob(>Q)"])
    # 确定p d q 
    (p, q) =(sm.tsa.arma_order_select_ic(y,max_ar=3,max_ma=3,ic='aic')['aic_min_order'])
    print(p,q)
    d=1
    arma_mod = ARMA(y,(p,d,q)).fit(disp=-1,method='mle')
    summary = (arma_mod.summary2(alpha=.05, float_format="%.8f"))
    print(summary)
    # 检验arma模型残差和白噪声
    arma_mod = ARMA(y,(p,d,q)).fit(disp=-1,method='mle')
    resid = arma_mod.resid
    t=sm.tsa.stattools.adfuller(resid)
    output=pd.DataFrame(index=['Test Statistic Value', "p-value", "Lags Used", "Number of Observations Used","Critical Value(1%)","Critical Value(5%)","Critical Value(10%)"],columns=['value'])
    output['value']['Test Statistic Value'] = t[0]
    output['value']['p-value'] = t[1]
    output['value']['Lags Used'] = t[2]
    output['value']['Number of Observations Used'] = t[3]
    output['value']['Critical Value(1%)'] = t[4]['1%']
    output['value']['Critical Value(5%)'] = t[4]['5%']
    output['value']['Critical Value(10%)'] = t[4]['10%']
    print(output)

    # 模型预测
    arma_model = sm.tsa.ARMA(y,(p,d,q)).fit(disp=-1,maxiter=100)
    predict_data = arma_model.predict(start=str(1979), end=str(2010+3), dynamic = False)
    plt.plot(x,y)
    plt.plot(x,predict_data)
    plt.show()
if __name__ == "__main__":
    plt.rcParams['font.family'] = ['sans-serif']
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    main()