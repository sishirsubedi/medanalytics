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

len(df_all_data)
df_all_data.head(2)
df_xdata = df_all_data.loc[:,'urea_n_min':'51250_var']
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

X= X_train
y = y_train.values

####################################################

import pandas as pd
import numpy as np
import statsmodels.api as sm
import  matplotlib.pylab as plt
from sklearn.cluster import KMeans,AgglomerativeClustering

def rbf(x, c, s):
    return np.exp(-1 / (2 * s**2) * (x-c)**2)


def kmeans(X, k):
    """Performs k-means clustering for 1D input

    Arguments:
        X {ndarray} -- A Mx1 array of inputs
        k {int} -- Number of clusters

    Returns:
        ndarray -- A kx1 array of final cluster centers
    """

    # randomly select initial clusters from input data
    clusters = np.random.choice(np.squeeze(X), size=k)
    prevClusters = clusters.copy()
    stds = np.zeros(k)
    converged = False

    while not converged:
        """
        compute distances for each cluster center to each point 
        where (distances[i, j] represents the distance between the ith point and jth cluster)
        """
        distances = np.squeeze(np.abs(X[:, np.newaxis] - clusters[np.newaxis, :]))

        # find the cluster that's closest to each point
        closestCluster = np.argmin(distances, axis=1)

        # update clusters by taking the mean of all of the points assigned to that cluster
        for i in range(k):
            pointsForCluster = X[closestCluster == i]
            if len(pointsForCluster) > 0:
                clusters[i] = np.mean(pointsForCluster, axis=0)

        # converge if clusters haven't moved
        converged = np.linalg.norm(clusters - prevClusters) < 1e-6
        prevClusters = clusters.copy()

    distances = np.squeeze(np.abs(X[:, np.newaxis] - clusters[np.newaxis, :]))
    closestCluster = np.argmin(distances, axis=1)

    clustersWithNoPoints = []
    for i in range(k):
        pointsForCluster = X[closestCluster == i]
        if len(pointsForCluster) < 2:
            # keep track of clusters with no points or 1 point
            clustersWithNoPoints.append(i)
            continue
        else:
            stds[i] = np.std(X[closestCluster == i])

    # if there are clusters with 0 or 1 points, take the mean std of the other clusters
    if len(clustersWithNoPoints) > 0:
        pointsToAverage = []
        for i in range(k):
            if i not in clustersWithNoPoints:
                pointsToAverage.append(X[closestCluster == i])
        pointsToAverage = np.concatenate(pointsToAverage).ravel()
        stds[clustersWithNoPoints] = np.mean(np.std(pointsToAverage))

    return clusters, stds


class RBFNet(object):
    """Implementation of a Radial Basis Function Network"""

    def __init__(self, k=2, lr=0.01, epochs=100, rbf=rbf, inferStds=True):
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
                a = np.array([self.rbf(X[i], c, s) for c, s, in zip(self.centers, self.stds)])
                F = a.T.dot(self.w) + self.b

                loss = (y[i] - F).flatten() ** 2
                #print('Loss: {0:.2f}'.format(loss[0]))

                # backward pass
                error = -(y[i] - F).flatten()

                # online update
                self.w = self.w - self.lr * a * error
                self.b = self.b - self.lr * error

    def predict(self, X):
        y_pred = []
        for i in range(X.shape[0]):
            a = np.array([self.rbf(X[i], c, s) for c, s, in zip(self.centers, self.stds)])
            F = a.T.dot(self.w) + self.b
            y_pred.append(F)
        return np.array(y_pred)

    def coefs(self):
        return self.w


# sample inputs and add noise

mixture = []

for i in range(X_train.shape[1]):
    rbfnet = RBFNet(lr=1e-2, k=10)
    X= X_train.iloc[:,i].values
    rbfnet.fit(X, y)
    coefs = rbfnet.coefs()
    mixture.append(coefs)

df_hidden_xdata = pd.DataFrame(mixture)

df_hidden_xdata.shape