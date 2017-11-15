import pandas as pd

from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression, Lasso

from scipy import stats

import statsmodels.api as sm

import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
import collections



#### icu stays
df_all_data = pd.read_csv("df_icu_admission_combine.csv",sep=',',header=0)#,index_col=0)
len(df_all_data)
df_all_data.head(2)
df_xdata = df_all_data.loc[:,'age':'urine_max']
df_n_xdata= (df_xdata-df_xdata.mean())/df_xdata.std()
df_n_xdata.mean()
df_n_xdata.head(20)

y = df_all_data['readmit']
y.shape
xmat = df_n_xdata
xmat.shape

### feature selection RFE

lr = LinearRegression()

aic =[]
bic =[]
rs = []
for k in range(1,31):

    rfe = RFE(lr, k)
    rfe.fit(xmat,y)

    xmat_filt = xmat.iloc[:, [x for x,d in enumerate(rfe.ranking_) if d ==1]]

    model = sm.OLS(y, xmat_filt)
    results = model.fit()

    aic.append(results.aic)
    bic.append(results.bic)
    rs.append(results.rsquared_adj)

plt.plot(range(1,31),aic,'ro-',label ='aic')
plt.plot(range(1,31),bic,'bo-',label ='bic')
plt.legend()


plt.plot(range(1,31),rs,'go-',label='rsq')
plt.legend()



#### LASSO regression

clf = Lasso(alpha=0.01)
clf.fit(xmat, y)
clf.coef_


#######