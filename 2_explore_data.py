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


############# cannot do chitest for feature selection because of negative values as this is centered
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

# plt.plot(llf,'ro-',label ='aic')
# plt.legend()

# McFadden’s pseudo-R-squared
plt.plot(rs,'go-',label='rsq')
plt.legend()
plt.xlabel('Number of features in the model (logistic regression)')
plt.ylabel('McFaddens pseudo R-squared')

# '''
# http://thestatsgeek.com/2014/02/08/r-squared-in-logistic-regression/
# McFadden's R squared is defined as 1-l_mod/l_null, where
# l_mod is the log likelihood value for the fitted model
# and l_null is the log likelihood for the null model which
# includes only an intercept as predictor (so that every
# individual is predicted the same probability of 'success').
# '''
#######################################################################################

  ### now select top 25 candidates



model = LogisticRegression()

k=25
rfe = RFE(model, k)
rfe = rfe.fit(X_train,y_train )

xmat_filt = X_train.iloc[:, [x for x,d in enumerate(rfe.ranking_) if d ==1]]

xmat_filt_test = X_test.iloc[:, [x for x,d in enumerate(rfe.ranking_) if d ==1]]



########################## feature selection using lasso regularization



C = [10, 1, .1, .01,0.001]
C = np.arange(0.001, 1.0, 0.01)


stat =  []
for c in C:
    clf = LogisticRegression(penalty='l1', C=c)
    clf.fit(X_train, y_train)
    print('C:', c)
    print('Coefficient of each feature:', clf.coef_)
    print('Training accuracy:', clf.score(X_train, y_train))
    print('Test accuracy:', clf.score(X_test, y_test))
    print('')

    stat.append([c,np.count_nonzero(clf.coef_[0]),clf.score(X_train, y_train),clf.score(X_test, y_test)])

cvals = [x[0] for x in stat]
coefs = [x[1] for x in stat]
train_ac = [x[2] for x in stat]
test_ac = [x[3] for x in stat]
plt.plot(cvals,coefs,'ro-',label='non-zero coefs')
plt.ylabel('Number of non zero coefficients')
plt.xlabel('C-value:Inverse of regularization strength')
plt.legend()
plt.plot(cvals,train_ac,'ro-',label='train_acc')
plt.plot(cvals,test_ac,'bo-',label='test_acc')
plt.ylabel('Accuracy score')
plt.xlabel('C-value:Inverse of regularization strength')
plt.legend()



c=0.5
clf = LogisticRegression(penalty='l1', C=c)
clf.fit(X_train, y_train)
print('C:', c)
print('Coefficient of each feature:', clf.coef_)
print('Training accuracy:', clf.score(X_train, y_train))
print('Test accuracy:', clf.score(X_test, y_test))
print('Number of non zero coefs:', np.count_nonzero(clf.coef_[0]))
print('')
