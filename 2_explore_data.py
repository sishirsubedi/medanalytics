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

