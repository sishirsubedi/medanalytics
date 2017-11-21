import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import random
import numpy as np
import math
import collections



df_all_data = pd.read_csv("cleaned_master.csv",sep=',',header=0)#,index_col=0)
df_all_data.head(2)
df_all_data.drop(['Unnamed: 0', 'Unnamed: 0.1'],axis=1,inplace=True)
df_all_data.head(2)
len(df_all_data)

df_all_data['admittime'].isnull().value_counts()
df_all_data['marital_status'].isnull().value_counts()
df_all_data['gender'].isnull().value_counts()
df_all_data['gender'].value_counts()
df_all_data['marital_status'].value_counts()


print df_all_data.T.apply(lambda x: x.nunique(), axis=1)


numeric_columns = df_all_data._get_numeric_data().columns
categorical_columns = list(set(df_all_data.columns)-set(numeric_columns))
df_all_data[categorical_columns].isnull().sum()*100/df_all_data.shape[0]
# here we see that 5% of marital status is missing
df_all_data['marital_status'].value_counts()




### fill missing marital status :::


grouped= df_all_data.groupby(['marital_status'])
#print type(bb)
names=[]
quantity=[]
nom_of_all=0
for status, group in grouped:
       names.append(status)
       quantity.append(len(group))
       nom_of_all=nom_of_all+len(group)

#print names
dist = [float(x) / nom_of_all for x in quantity]
#print dist

for i in range(0, len(df_all_data)):
       if pd.isnull(df_all_data.loc[i, 'marital_status'])==False:
               continue
       df_all_data.loc[i, 'marital_status']=np.random.choice(names, p=dist)


df_all_data['marital_status'].isnull().value_counts()


##########


df_all_data.fillna(df_all_data.mean(),inplace=True)

df_all_data.isnull().sum()*100/df_all_data.shape[0]

df_all_data.shape


#########################################################################


all_patients = df_all_data.iloc[:,0].values

# get the frequency table for multiple icu patient id record
#collections.Counter(all_patients)
patient_freq ={}
for x in all_patients:
    if x not in patient_freq:
        patient_freq[x] = 1
    else:
        patient_freq[x] += 1

plt.plot(range(1,len(patient_freq)+1),patient_freq.values())
plt.xlabel('Patients')
plt.ylabel('Times ICU readmitted')
plt.show()

len(patient_freq)


## get patients with two readmission
icu_admissionid_times_2 = [k for k in patient_freq if patient_freq[k] >=2]
len(icu_admissionid_times_2)


final_patients =[]
icu_admission_times_2 =[]
for id in icu_admissionid_times_2:
    row = df_all_data.loc[df_all_data['subject_id']==id]

    first_admit = datetime.strptime(row['admittime'].values[0], '%Y-%m-%d%H:%M:%S')
    first_discharge = datetime.strptime(row['dischtime'].values[0], '%Y-%m-%d%H:%M:%S')

    second_admit = datetime.strptime(row['admittime'].values[1], '%Y-%m-%d%H:%M:%S')
    second_discharge = datetime.strptime(row['dischtime'].values[1], '%Y-%m-%d%H:%M:%S')

    ## make sure its not internal ICU transfer
    same_day = abs((first_admit - second_admit).days)
    if same_day == 0: continue


    days = abs((second_admit - first_discharge).days)

    age = int(row['age'].values[0])

    if age == 0: continue

    if age > 100 :
        age= 90
        row['age'].values[0] = 90
        row['age'].values[1] = 90

    if days <31 :#and row['deathtime'].isnull().any(): ?? what to do with dead patient ## dont worry if patient dies during second icu visit
        ## check to get the first admission data for analysis
        if (second_admit - first_admit).days > 0:
            icu_admission_times_2.append(row.values[0])
        else :
            icu_admission_times_2.append(row.values[0])
        final_patients.append([id,age,days])

len(final_patients)

patient_ids = [x[0]for x in final_patients]
len(patient_ids)

patient_days = [x[2]for x in final_patients]
p = {x:patient_days.count(x) for x in patient_days}
plt.bar(p.keys(), p.values(), color='b')
#plt.boxplot(x,y)
plt.xlabel('Interval days for second ICU admission')
plt.ylabel('Patient count')
plt.show()

### make box plot of days




#
# from scipy.interpolate import interp1d
# f = interp1d(x, y)
# plt.plot(x, y, 'o', x, f(x))

patient_age = [x[1]for x in final_patients]
p = {x:patient_age.count(x) for x in patient_age}
plt.bar(p.keys(), p.values(), color='b')
plt.xlabel(' Age of patients with two ICU admission')
plt.ylabel('Patient count ')
plt.show()


df_icu_admission_times_2= pd.DataFrame(icu_admission_times_2,columns=df_all_data.columns.values)
df_icu_admission_times_2.head(2)

df_icu_admission_times_2.drop(['hadm_id','admittime','dischtime','first_careunit','last_careunit','insurance'], axis=1, inplace=True)
df_icu_admission_times_2.head()
# from sklearn import preprocessing
# le = preprocessing.LabelEncoder()
# le.fit(df_icu_admission_times_2['gender'])
# le.transform(df_icu_admission_times_2['gender'])
df_icu_admission_times_2['gender'].value_counts()
df_icu_admission_times_2['marital_status'].value_counts()

##### this is for label encoding ##########
cleanup_nums = {"gender": {"M":1.0, "F": 0.0},
                "marital_status" : {"MARRIED":1.0, "SINGLE":2.0,"WIDOWED":3.0, "DIVORCED":4.0, "SEPARATED":5.0,"UNKNOWN(DEFAULT)":6.0}}
df_icu_admission_times_2.replace(cleanup_nums, inplace=True)

################## this is for hot encoding

df_icu_admission_times_2.head(2)
df_icu_admission_times_2= pd.get_dummies(df_icu_admission_times_2,columns=['gender','marital_status'],drop_first=False)
df_icu_admission_times_2.head(2)

df_icu_admission_times_2.drop(['marital_status_UNKNOWN(DEFAULT)'], axis=1, inplace=True)

df_icu_admission_times_2['readmit'] = 1.0
df_icu_admission_times_2.head()
len(df_icu_admission_times_2)
df_icu_admission_times_2.to_csv("icu_admission_times_2.csv", index=False)



icu_admissionid_only_1 = [k for k in patient_freq if patient_freq[k] ==1]
len(icu_admissionid_only_1)
random.seed(1)

icu_admissionid_only_1 = random.sample(icu_admissionid_only_1, len(df_icu_admission_times_2))
len(icu_admissionid_only_1)

final_patients_control =[]
icu_admission_times_1 =[]
for id in icu_admissionid_only_1:
    row = df_all_data.loc[df_all_data['subject_id']==id]

    age = int(row['age'].values[0])

    if age == 0: continue

    if age > 100 :
        age = 90
        row['age'].values[0] = 90


    # if row['deathtime'].isnull().any():
    icu_admission_times_1.append(row.values[0])
    final_patients_control.append([id,age])


df_icu_admission_times_1= pd.DataFrame(icu_admission_times_1,columns=df_all_data.columns.values)
len(df_icu_admission_times_1)

patient_ids_control = [x[0]for x in final_patients_control]
len(patient_ids_control)


patient_age_c = [x[1]for x in final_patients_control]
pc = {x:patient_age_c.count(x) for x in patient_age_c}
plt.bar(pc.keys(), pc.values(), color='g')
plt.xlabel('Age of patients with one ICU admission ')
plt.ylabel('Patient count ')
plt.show()

## try age overlay


df_icu_admission_times_1.head(2)
df_icu_admission_times_1.drop(['hadm_id','admittime','dischtime','first_careunit','last_careunit','insurance'], axis=1, inplace=True)
df_icu_admission_times_1.head()


df_icu_admission_times_1.head(2)
df_icu_admission_times_1= pd.get_dummies(df_icu_admission_times_1,columns=['gender','marital_status'],drop_first=False)
df_icu_admission_times_1.head(2)

df_icu_admission_times_1.drop(['marital_status_UNKNOWN(DEFAULT)'], axis=1, inplace=True)

df_icu_admission_times_1['readmit'] = 0.0
df_icu_admission_times_1.head()
len(df_icu_admission_times_1)
df_icu_admission_times_1.to_csv("icu_admission_times_1.csv", index=False)




frames = [df_icu_admission_times_1,df_icu_admission_times_2]

df_icu_admission_combine = pd.concat(frames)

len(df_icu_admission_combine)


## shuffle rows
df_icu_admission_combine = df_icu_admission_combine.sample(frac=1).reset_index(drop=True)

df_icu_admission_combine.to_csv("df_icu_admission_combine.csv", index=False)



#######################################################################
#
#
# ####
# df_icu_admission_times_1['gender'].value_counts()
# df_icu_admission_times_1['marital_status'].value_counts()
# cleanup_nums = {"gender": {"M": 1, "F": 0},
#                 "marital_status" : {"MARRIED": 1 , "SINGLE":2 ,"WIDOWED":3 , "DIVORCED":4 , "SEPARATED":5 ,"UNKNOWN(DEFAULT)": 6}}
# df_icu_admission_times_1.replace(cleanup_nums, inplace=True)
# df_icu_admission_times_1['readmit'] = 0.0
# #df_icu_admission_times_1['albumin_min']
# df_icu_admission_times_1.fillna(df_icu_admission_times_1.mean(),inplace=True)
# df_icu_admission_times_1.head()
# df_icu_admission_times_1.to_csv("icu_admission_times_1.csv", index=False)
#
# frames = [df_icu_admission_times_1,df_icu_admission_times_2]
#
# df_icu_admission_combine = pd.concat(frames)
#
# len(df_icu_admission_combine)
#
#
# ## shuffle rows
# df_icu_admission_combine = df_icu_admission_combine.sample(frac=1).reset_index(drop=True)
#
# df_icu_admission_combine.to_csv("df_icu_admission_combine.csv", index=False)
