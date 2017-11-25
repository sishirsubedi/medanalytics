import pandas as pd
import numpy as np
import statsmodels.api as sm
import  matplotlib.pylab as plt
from sklearn.cluster import KMeans,AgglomerativeClustering

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


df_all_data = pd.read_csv("df_icu_admission_combine.csv",sep=',',header=0)#,index_col=0)
len(df_all_data)
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
y_train = pd.DataFrame(y_train)
y_test = pd.DataFrame(y_test)


X_train.shape
y_train.shape



k=1


model = LogisticRegression()
rfe = RFE(model, k)
rfe = rfe.fit(X_train.values,y_train.values )
logr_ranking =[]
for x,d in zip(rfe.ranking_,X_train.columns):
    logr_ranking.append([d,x])
logr_ranking = pd.DataFrame(logr_ranking,columns=['features','logr'])
logr_ranking.sort_values('logr',inplace=True)


todrop = logr_ranking.iloc[10: ,0]
df_all_data.drop(todrop, axis=1, inplace=True)
df_all_data.shape

df_all_data.to_csv("test_data.csv",index=False)

len(df_all_data)
df_all_data.head(2)
df_xdata = df_all_data.loc[:,'urea_n_min':'51006_min']
df_n_xdata= (df_xdata-df_xdata.mean())/df_xdata.std()
df_n_xdata.mean()
df_n_xdata.head(20)


df_ydata = df_all_data.loc[:,'readmit']
df_ydata.head(2)




### split train test

X_train, X_test, y_train, y_test = train_test_split( df_n_xdata, df_ydata, test_size=0.33, random_state=42)
y_train = pd.DataFrame(y_train)
y_test = pd.DataFrame(y_test)


X_train.shape
y_train.shape


####################################################

import pandas as pd
import numpy as np
import statsmodels.api as sm
import  matplotlib.pylab as plt
from sklearn.cluster import KMeans,AgglomerativeClustering

import math

def sigmoid(x):
  return 1 / (1 + math.exp(-x))


def rbf(x, mu, s):

    var = sum(s)/len(s)

    return np.exp(-1 / (2 * var) * np.sum((x-mu)**2))


def kmeans(xdata, k):

    clust_num = k
    # kmeans = KMeans(n_clusters=clust_num, random_state=0).fit(xdata)
    hc = AgglomerativeClustering(n_clusters=clust_num).fit(xdata)
    xdata= pd.DataFrame(xdata)
    xdata['clust'] = hc.labels_
    xdata.head(20)

    centers =[]
    stds =[]

    for c in range(k):
        xdata_1 = xdata.loc[xdata['clust'] == c]
        centers.append(xdata_1.iloc[:, 0:xdata.shape[1]-1].mean())
        stds.append(xdata_1.iloc[:, 0:xdata.shape[1] - 1].var())

    return centers, stds


############# learning
class RBFNet(object):
    """Implementation of a Radial Basis Function Network"""

    def __init__(self, k=2, lr=0.001, epochs=1000, rbf=rbf, inferStds=True):
        self.k = k
        self.lr = lr
        self.epochs = epochs
        self.rbf = rbf
        self.inferStds = inferStds

        self.w = np.random.randn(k)
        self.b = np.random.randn(1)

    def fit(self, X, y):
        if self.inferStds:
            # compute stds from data
            self.centers, self.stds = kmeans(X, self.k)
        else:
            # use a fixed std
            self.centers, _ = kmeans(X, self.k)
            dMax = max([np.abs(c1 - c2) for c1 in self.centers for c2 in self.centers])
            self.stds = np.repeat(dMax / np.sqrt(2*self.k), self.k)

        # training
        for epoch in range(self.epochs):
            for i in range(X.shape[0]):
                # forward pass
                a=[]
                for cen in range(len(self.centers)):
                    a.append(rbf(X[i], self.centers[cen].values, self.stds[cen].values))
                a = np.array(a)

                F = a.T.dot(self.w) + self.b

                #yi = np.array([0.5 if y[i] == 0.0 else 1.0])
                loss = -(y[0] - F) .flatten()
                #print('Loss: {0:.2f}'.format(loss[0]))

                # backward pass
                # error = -(y[i] - F).flatten()

                # online update
                self.w = self.w - self.lr * a * loss[0]
                self.b = self.b - self.lr * loss[0]
            print self.w, self.b

    def predict(self, X):
        y_pred = []

        for i in range(X.shape[0]):
            # forward pass
            a=[]
            for cen in range(len(self.centers)):
                a.append(rbf(X[i], self.centers[cen].values, self.stds[cen].values))
            a = np.array(a)

            F = a.T.dot(self.w) + self.b
            yi = np.array([0.0 if sigmoid(F) < 0.5 else 1.0])
            y_pred.append(yi)


        return np.array(y_pred)

    def coefs(self):
        return self.w







rbfnet = RBFNet(lr=1e-2, k=2,epochs=1000)
X= X_train.values
print X.shape
y = y_train.values
rbfnet.fit(X, y)
yhat = rbfnet.predict(X)

match = 0.0
for m in range(0, len(yhat)):
    if yhat[m] == y[m]: match += 1

match = match / len(yhat)

print match



 ############### use LM for optimization

class RBFNet_LM(object):
    """Implementation of a Radial Basis Function Network"""

    def __init__(self, k=2, rbf=rbf, inferStds=True):
        self.k = k
        self.rbf = rbf
        self.inferStds = inferStds


    def fit(self, X, y):
        if self.inferStds:
            # compute stds from data
            self.centers, self.stds = kmeans(X, self.k)
        else:
            # use a fixed std
            self.centers, _ = kmeans(X, self.k)
            dMax = max([np.abs(c1 - c2) for c1 in self.centers for c2 in self.centers])
            self.stds = np.repeat(dMax / np.sqrt(2*self.k), self.k)

        # training

        xmat = []
        for i in range(X.shape[0]):
            # forward pass
            a=[]
            for cen in range(len(self.centers)):
                a.append(rbf(X[i], self.centers[cen].values, self.stds[cen].values))
            temp =[1.0]

            for x in a : temp.append(x)
            xmat.append(temp)
            print temp
        return xmat


rbfnet = RBFNet_LM(k=2)
X= X_train.values
print X.shape
y = y_train.values
xmat = rbfnet.fit(X, y)

xmat = np.array(xmat)

logit = sm.Logit(y_train, xmat)
results = logit.fit()

yhat = pd.DataFrame(results.predict(xmat), columns=['predict'])

yhat['predict'] = yhat['predict'].apply(lambda x: 0.0 if x < 0.5 else 1.0)

match = 0.0
for m in range(0, len(yhat)):
    if yhat.iloc[m].values == y_train.iloc[m].values: match += 1

match = match / len(yhat)
print match