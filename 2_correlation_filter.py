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
df_xdata = df_all_data.loc[:,'age':'marital_status_WIDOWED']
df_n_xdata= (df_xdata-df_xdata.mean())/df_xdata.std()
df_n_xdata.mean()
df_n_xdata.head(20)

corr_matrix = df_n_xdata.corr()
corr_matrix.shape

sns.heatmap(corr_matrix,xticklabels=corr_matrix.columns,yticklabels=corr_matrix.columns)


cormat_melted = []
for i in range(len(corr_matrix)):
    f1 = corr_matrix.columns[i]
    for j in range(i,len(corr_matrix)):
        f2 = corr_matrix.columns[j]
        cormat_melted.append([f1, f2, corr_matrix.iloc[i,j]])
cormat_melted = pd.DataFrame(cormat_melted,columns=['f1','f2','values'])
cormat_melted.head(5)
cormat_melted_filt = cormat_melted.loc[(cormat_melted['values']>=0.75) & (cormat_melted['values'] !=1.0)]
todrop = set(cormat_melted_filt['f2'])
len(todrop)


df_all_data.drop(todrop, axis=1, inplace=True)

df_all_data.head(2)
df_all_data.shape
#df_all_data.to_csv("df_icu_admission_combine_corfilt.csv", index=False)

df_xdata = df_all_data.loc[:,'age':'marital_status_WIDOWED']
df_n_xdata= (df_xdata-df_xdata.mean())/df_xdata.std()
df_n_xdata.mean()
df_n_xdata.head(20)

corr_matrix = df_n_xdata.corr()
corr_matrix.shape

sns.heatmap(corr_matrix,xticklabels=corr_matrix.columns,yticklabels=corr_matrix.columns)


################### describe data

df_all_data.describe()

print pd.crosstab(df_all_data['readmit'], df_all_data['gender_M'], rownames=['readmit'])

