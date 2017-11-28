import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import random
import numpy as np
import math
import collections
import seaborn as sns
import statsmodels.api as sm
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from xgboost.sklearn import XGBClassifier


df_all_data = pd.read_csv("df_icu_admission_combine_corfilt.csv",sep=',',header=0)#,index_col=0)
len(df_all_data)
df_all_data.shape
df_all_data.head(2)
df_xdata = df_all_data.loc[:,'age':'marital_status_WIDOWED']
df_n_xdata= (df_xdata-df_xdata.mean())/df_xdata.std()
df_n_xdata.mean()
df_n_xdata.head(20)


df_ydata = df_all_data.loc[:,'readmit']
df_ydata.head(2)




### split train test

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split( df_n_xdata, df_ydata, test_size=0.33, random_state=42)
print "Ratio in training", str(sum(y_train==1.0) * 100 /len(y_train))
print "Ratio in test", str(sum(y_test==1.0) * 100 /len(y_test))


y_train = pd.DataFrame(y_train)
X_train = pd.DataFrame(X_train)

######################### feature selection using rfe #################

# McFadden’s pseudo-R-squared

# '''
# http://thestatsgeek.com/2014/02/08/r-squared-in-logistic-regression/
# McFadden's R squared is defined as 1-l_mod/l_null, where
# l_mod is the log likelihood value for the fitted model
# and l_null is the log likelihood for the null model which
# includes only an intercept as predictor (so that every
# individual is predicted the same probability of 'success').
# '''

################# get feature ranking

k=1

model = LogisticRegression()
rfe = RFE(model, k)
rfe = rfe.fit(X_train,y_train )
logr_ranking =[]
for x,d in zip(rfe.ranking_,X_train.columns):
    logr_ranking.append([d,x])
logr_ranking = pd.DataFrame(logr_ranking,columns=['features1','logr'])
logr_ranking.sort_values('features1',inplace=True)



model = XGBClassifier()
rfe = RFE(model, k)
rfe = rfe.fit(X_train,y_train )
xgboost_ranking =[]
for x,d in zip(rfe.ranking_,X_train.columns):
    xgboost_ranking.append([d,x])
xgboost_ranking = pd.DataFrame(xgboost_ranking,columns=['features2','xgboost'])
xgboost_ranking.sort_values('features2',inplace=True)


model = LinearSVC()
rfe = RFE(model, k)
rfe = rfe.fit(X_train,y_train )
lsvc_ranking =[]
for x,d in zip(rfe.ranking_,X_train.columns):
    lsvc_ranking.append([d,x])
lsvc_ranking = pd.DataFrame(lsvc_ranking,columns=['features3','lsvc'])
lsvc_ranking.sort_values('features3',inplace=True)


logr_ranking.head(1)
xgboost_ranking.head(1)
lsvc_ranking.head(1)

xgboost_ranking.to_csv("xgboost_ranking.csv", index=False)
lsvc_ranking.to_csv("lsvc_ranking.csv", index=False)
logr_ranking.to_csv("logr_ranking.csv", index=False)

xgboost_ranking = pd.read_csv("xgboost_ranking.csv")
xgboost_ranking.head(2)
lsvc_ranking =  pd.read_csv("lsvc_ranking.csv")
lsvc_ranking.head(2)
logr_ranking =   pd.read_csv("logr_ranking.csv")
logr_ranking.head(2)

feature_sum = logr_ranking['logr']+ xgboost_ranking['xgboost']+lsvc_ranking['lsvc']
df_ranked =  pd.concat([logr_ranking['features1'],feature_sum],axis=1)
df_ranked.head(2)
len(df_ranked)
df_ranked.sort_values(0,inplace=True)
df_ranked.head(20)


############# train and cross validation

X_train0, X_cv, y_train0, y_cv = train_test_split( X_train, y_train, test_size=0.20, random_state=42)
print  sum(y_train0['readmit']==1.0) * 100 /len(y_train0)
print  sum(y_cv['readmit']==1.0) * 100 /len(y_cv)


############### select best k
df_ranked.head(2)

ranked_list = list(df_ranked['features1'])

X_train0.head(2)
X_train0_sorted = X_train0[ranked_list]
X_train0_sorted.head(2)


X_cv.head(2)
X_cv_sorted = X_cv[ranked_list]
X_cv_sorted.head(2)

accuracy = []

for k in range(1,121):

    x_train_filt = X_train0_sorted.iloc[:,0:k]

    logit = sm.Logit(y_train0, x_train_filt)

    results = logit.fit()

    yhat = pd.DataFrame(results.predict(X_cv_sorted.iloc[:,0:k]), columns=['predict'])


    yhat['predict'] = yhat['predict'].apply(lambda x: 0.0 if x < 0.5 else 1.0)

    match = 0.0
    for m in range(0, len(yhat)):
        if yhat.iloc[m].values == y_cv.iloc[m].values: match += 1

    match = match / len(yhat)

    accuracy.append(match)

plt.rcParams.update({'font.size': 15})
plt.plot(accuracy,'bo-',label='logisticR')
plt.legend()
#plt.xlim(range(1,100))
plt.xlabel('Number of features in the model')
plt.ylabel('Accuracy on validation set')
plt.title('Feature selection using ranking on 3 models with recursive feature elimination')




###################


#
# model = XGBClassifier()
# xgboost_rs = []
# for k in range(1,30):
#
#     rfe = RFE(model, k)
#     rfe = rfe.fit(X_train,y_train )
#
#     xmat_filt = X_train.iloc[:, [x for x,d in enumerate(rfe.ranking_) if d ==1]]
#
#     logit = sm.Logit(y_train, xmat_filt)
#     results = logit.fit()
#
#     yhat = pd.DataFrame(results.predict(xmat_filt), columns=['predict'])
#
#     yhat['predict'] = yhat['predict'].apply(lambda x: 0.0 if x < 0.5 else 1.0)
#
#     match = 0.0
#     for m in range(0, len(yhat)):
#         if yhat.iloc[m].values == y_train.iloc[m].values: match += 1
#
#     match = match / len(yhat)
#
#     # xgboost_rs.append(results.prsquared)
#
#     xgboost_rs.append(match)


# model = LinearSVC()
# lsv_rs = []
# for k in range(1,30):
#
#     rfe = RFE(model, k)
#     rfe = rfe.fit(X_train,y_train )
#
#     xmat_filt = X_train.iloc[:, [x for x,d in enumerate(rfe.ranking_) if d ==1]]
#
#     logit = sm.Logit(y_train, xmat_filt)
#     results = logit.fit()
#
#     yhat = pd.DataFrame(results.predict(X_cv.iloc[:, [x for x,d in enumerate(rfe.ranking_) if d ==1]]), columns=['predict'])
#
#     yhat['predict'] = yhat['predict'].apply(lambda x: 0.0 if x < 0.5 else 1.0)
#
#     match = 0.0
#     for m in range(0, len(yhat)):
#         if yhat.iloc[m].values == y_train.iloc[m].values: match += 1
#
#     match = match / len(yhat)
#
#     lsv_rs.append(match)





k=1
model = LogisticRegression()
rfe = RFE(model, k)
rfe = rfe.fit(X_train0,y_train0)


logr_rs = []
for k in range(1,50):

    xmat_filt = X_train0.iloc[:, [x for x,d in enumerate(rfe.ranking_) if d in range(1,k+1)]]

    logit = sm.Logit(y_train0, xmat_filt)
    results = logit.fit()

    yhat = pd.DataFrame(results.predict(X_cv.iloc[:, [x for x,d in enumerate(rfe.ranking_) if d in range(1,k+1)]]),columns=['predict'])

    yhat['predict'] = yhat['predict'].apply(lambda x: 0.0 if x < 0.5 else 1.0 )

    match =0.0
    for m in range(0,len(yhat)):
        if yhat.iloc[m].values == y_cv.iloc[m].values : match += 1

    match = match/len(yhat)

    #logr_rs.append(results.prsquared)

    logr_rs.append(match)


plt.rcParams.update({'font.size': 15})
# plt.plot(xgboost_rs,'go-',label='xgboost')
# plt.plot(lsv_rs,'ro-',label='linearSV')
plt.plot(logr_rs,'bo-',label='logisticR')
plt.legend()
plt.xlim([1,30])
plt.xlabel('Number of features in the model')
plt.ylabel('Accuracy on validation set')
plt.title('Feature selection using recursive feature elimination')



#######################################################################################

### now select top 25 candidates and test logistic regression model



model = LogisticRegression()

k=25
rfe = RFE(model, k)
rfe = rfe.fit(X_train,y_train )

xmat_filt = X_train.iloc[:, [x for x,d in enumerate(rfe.ranking_) if d ==1]]

xmat_filt_test = X_test.iloc[:, [x for x,d in enumerate(rfe.ranking_) if d ==1]]



########################## feature selection using lasso regularization



#C = [10, 1, .1, .01,0.001]
C = np.arange(0.01, 1.0, 0.01)


stat =  []
for c in C:
    clf = LogisticRegression(penalty='l1', C=c)
    clf.fit(X_train0, y_train0)

    xmat_filt = X_train0.iloc[:, [x for x in np.flatnonzero(clf.coef_[0])]]

    if xmat_filt.shape[1]==0:continue
    logit = LogisticRegression()
    results = logit.fit( xmat_filt.values,y_train0.values)

    match = logit.score(xmat_filt, y_train0)

    #logr_rs.append(results.prsquared)


    #stat.append([c,np.count_nonzero(clf.coef_[0]),clf.score(X_train, y_train),clf.score(X_test, y_test),results.prsquared])

    stat.append([c,np.count_nonzero(clf.coef_[0]),match,clf.score(X_train0, y_train0),clf.score(X_cv, y_cv),clf.score(X_train, y_train)])

cvals = [x[0] for x in stat]
coefs = [x[1] for x in stat]
rsq = [x[2] for x in stat]
train_ac = [x[3] for x in stat]
valid_ac = [x[4] for x in stat]
test_ac = [x[5] for x in stat]


plt.plot(cvals,rsq,'bo-',label='')
plt.ylabel('Accuracy on training set')
plt.xlabel('C-value:Inverse of regularization strength')
plt.legend()

plt.plot(cvals,coefs,'ro-')
plt.ylabel('Number of non zero coefficients')
plt.xlabel('C-value:Inverse of regularization strength')
plt.legend()


plt.plot(cvals,train_ac,'go-',label='train_acc')
plt.plot(cvals,valid_ac,'bo-',label='validation_acc')
plt.plot(cvals,test_ac,'ro-',label='test_acc')
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
