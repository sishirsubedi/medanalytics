import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import random
import numpy as np
import math
import collections
import seaborn as sns


df_all_data = pd.read_csv("df_icu_admission_combine.csv",sep=',',header=0)#,index_col=0)
len(df_all_data)
df_all_data.head(2)
df_xdata = df_all_data.loc[:,'age':'urine_max']
df_n_xdata= (df_xdata-df_xdata.mean())/df_xdata.std()
df_n_xdata.mean()
df_n_xdata.head(20)


df_ydata = df_all_data.loc[:,'readmit']
df_ydata.head(2)




### split train test

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split( df_n_xdata, df_ydata, test_size=0.33, random_state=42)

X_train.shape
y_train.shape






#### correlation selection
# calculate the correlation matrix
corr = df_xdata.corr()
# plot the heatmap
sns.heatmap(corr,xticklabels=corr.columns,yticklabels=corr.columns)
df_all_data.drop(['urea_n_min','urea_n_max','platelets_min','platelets_max','temp_max'], axis=1, inplace=True)


df_n_xdata.head(2)



################### describe data

df_all_data.describe()

print pd.crosstab(df_all_data['readmit'], df_all_data['gender'], rownames=['readmit'])


 ############# cannot to chitest for feature selection because of negative values as this is centered
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
# load data
X = np.array(df_n_xdata,dtype=float)
print X.shape
Y = np.array(df_ydata,dtype=float)
print Y.shape
# feature extraction
test = SelectKBest(score_func=chi2, k=5)
fit = test.fit(X, Y)
# summarize scores
np.set_printoptions(precision=3)
print(fit.scores_)
features = fit.transform(X)
# summarize selected features
print(features[0:5,:])
####################################################################



######################### feature selection using rfe #################

from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm

model = LogisticRegression()


llf =[]
rs = []
for k in range(1,34):

    rfe = RFE(model, k)
    rfe = rfe.fit(X_train,y_train )

    xmat_filt = X_train.iloc[:, [x for x,d in enumerate(rfe.ranking_) if d ==1]]

    logit = sm.Logit(y_train, xmat_filt)
    results = logit.fit()

    #results.summary()

    llf.append(results.llf)
    rs.append(results.prsquared)

plt.plot(llf,'ro-',label ='aic')
plt.legend()


plt.plot(rs,'go-',label='rsq')
plt.legend()
plt.xlabel('Number of features in the model (logistic regression)')
plt.ylabel('Rsquared')


#######################################################################################






