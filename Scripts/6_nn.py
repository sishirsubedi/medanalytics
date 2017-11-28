#http://www.wildml.com/2015/09/implementing-a-neural-network-from-scratch/

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math


df_data = pd.read_csv('test_data.csv',header=0)
df_data.head(3)
df_data = df_data.loc[:,'urea_n_min':]
