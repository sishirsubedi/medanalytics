#https://github.com/jkim0120/Naive-Bayes-Python/blob/master/Naive-Bayes.ipynb

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
from random import randrange


def partition_data(dataset, ratio):
    train_size = int(len(dataset) * ratio)
    test_set = list(dataset)
    train_set = []

    while len(train_set) < train_size:
        index = randrange(len(test_set))
        train_set.append(test_set.pop(index))

    return [train_set, test_set]


def group_by_class(dataset):
    klass_map = {}
    for line in dataset:
        klass = int(line[-1])
        if klass not in klass_map:
            klass_map[klass] = []
        klass_map[klass].append(line[:-1])
    return klass_map


def mean(n):
    return sum(n) / float(len(n))

def stdev(n):
    average = mean(n)
    return math.sqrt(sum([pow(x - average, 2) for x in n]) / float(len(n) - 1))


def gauss(x, mean, stdev):
    ex = math.exp(-(math.pow(x - mean, 2) / (2 * math.pow(stdev, 2))))
    return (1 / (math.sqrt(2 * math.pi) * stdev)) * ex


def format_calc(t):
    return (mean(t), stdev(t))

def prepare_data(dataset):
    summary = {}
    for klass, data_points in dataset.iteritems():
        summary[klass] = []
        for i in range(0,len(data_points[0])):
            summary[klass].append(format_calc([x[i] for x in data_points]))
    return summary

def predict(summary_set, data_point):
    probabilities = {}
    for klass, summary in summary_set.iteritems():
        probabilities[klass] = 1
        for i in xrange(len(summary)):
            mean, stdev = summary[i]
            probabilities[klass] *= gauss(data_point[i], mean, stdev)
    # this is pythonic way of finding key with max value
    return max(probabilities.iterkeys(), key=(lambda key: probabilities[key]))


def get_accuracy(summary_set, test_set):
    correct_count = 0
    for test_point in test_set:
        if test_point[-1] == predict(summary_set, test_point):
            correct_count += 1
    return correct_count / float(len(test_set)) * 100



df_data = pd.read_csv('test_data.csv',header=0)
df_data.head(3)
df_data = df_data.loc[:,'urea_n_min':]
train_set, test_set = partition_data([r.tolist() for i,r in df_data.iterrows()], 0.70)
classified_set = group_by_class(train_set)
summary_set = prepare_data(classified_set)
accuracy = get_accuracy(summary_set, test_set)
print('The Naive-Bayes Model yields {0}% accuracy').format(round(accuracy, 2))



## test scikit example http://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html

from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf.fit(df_data.loc[:,'urea_n_min':'51250_var'], df_data.loc[:,'readmit'])
yhat = pd.DataFrame(clf.predict(df_data.loc[:,'urea_n_min':'51250_var']), columns=['predict'])


y_train = pd.DataFrame(df_data.loc[:,'readmit'])

match = 0.0
for m in range(0, len(yhat)):
    if yhat.iloc[m].values == y_train.iloc[m].values: match += 1

match = match / len(yhat)
print match
