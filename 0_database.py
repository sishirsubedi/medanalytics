# Import libraries
import pandas as pd
import psycopg2
import matplotlib.pyplot as plt
import getpass

# Create a database connection
user = 'postgres'
host = 'localhost'
dbname = 'mimic'
schema = 'mimiciii'

# Connect to the database
con = psycopg2.connect(dbname=dbname, user=user, host=host,
                       password=getpass.getpass(prompt='Password:'.format(user)))
cur = con.cursor()
cur.execute('SET search_path to {}'.format(schema))

# Get length of stay from the icustays table
query = \
"""
select SUBJECT_ID,le.ITEMID,dl.label, avg(valuenum) as mean_val, max(valuenum) as max_val,  min(valuenum) as min_val, variance(valuenum) as var_val
    from labevents le
inner join D_LABITEMS dl 
	on  dl.ITEMID=le.ITEMID
 where hadm_id is not null
    group by SUBJECT_ID, le.ITEMID, dl.label
   having avg(valuenum) >0;
"""

data_le = pd.read_sql_query(query,con)
data_le.head()

len(data_le.subject_id.unique())

df1 =data_le.pivot(index='subject_id',columns='itemid', values = 'mean_val')
df2 =data_le.pivot(index='subject_id',columns='itemid', values = 'max_val')
df3 =data_le.pivot(index='subject_id',columns='itemid', values = 'min_val')
df4 =data_le.pivot(index='subject_id',columns='itemid', values = 'var_val')
df1.columns = [str(col) + '_mean' for col in df1.columns]
df2.columns = [str(col) + '_max' for col in df2.columns]
df3.columns = [str(col) + '_min' for col in df3.columns]
df4.columns = [str(col) + '_var' for col in df4.columns]
df1=df1.reset_index()
df2=df2.reset_index()
df3=df3.reset_index()
df4=df4.reset_index()


df_master = pd.merge(df1,df2, on ='subject_id')
df_master = pd.merge(df_master,df3, on ='subject_id')
df_master = pd.merge(df_master,df4, on ='subject_id')

df_all_data = pd.read_csv('all_data.csv');

df_master_all = pd.merge(df_all_data,df_master, on = 'subject_id')
df_master_all.to_csv('df_master_all.csv');

dropdf=pd.read_csv('df_master_all.csv');
nan_cols =dropdf.isnull().sum()*100/42734
nan_cols_sorted=nan_cols.sort_values(ascending=False)
nan_cols_sorted

nan_cols_sorted[nan_cols_sorted<25]
df_2 = dropdf.drop(list(nan_cols_sorted[nan_cols_sorted>25].index),axis=1)
df_3 = df_2.drop(['platelets_min','platelets_max',\
                 'platelets_mean','magnesium_max','calcium_min'],axis=1)

df_3.to_csv('cleaned_master.csv')



import pandas as pd
raw_data = {
        'subject_id': ['1', '2', '3', '4', '5'],
        'first_name': ['Alex', 'Amy', 'Allen', 'Alice', 'Ayoung'],
        'last_name': ['Anderson', 'Ackerman', 'Ali', 'Aoni', 'Atiches']}
df_a = pd.DataFrame(raw_data, columns = ['subject_id', 'first_name', 'last_name'])
df_a

raw_data = {
        'subject_id': ['1', '2', '3', '4', '5', '7', '8', '9', '10', '11'],
        'test_id': [51, 15, 15, 61, 16, 14, 15, 1, 61, 16]}
df_n = pd.DataFrame(raw_data, columns = ['subject_id','test_id'])
df_n

newd = pd.merge(df_a, df_n, on='subject_id')
newd