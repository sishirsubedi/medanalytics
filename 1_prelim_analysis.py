import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
import collections



#### icu stays

df_icu_admissions  = pd.read_csv("ICUSTAYS.csv",sep=',',header=0)
df_icu_admissions.head(10)
data = [data for index,data in df_icu_admissions.iteritems()]
subject_id = data[1]


# get the frequency table for multiple icu patient id record
p_freq ={}
for x in subject_id:
    if x not in p_freq:
        p_freq[x] = 1
    else:
        p_freq[x] += 1
# plt.plot(range(len(p_freq)),p_freq.values())
# plt.xlabel('Patients')
# plt.ylabel('Times ICU readmitted')
# plt.show()



# get patient id with only 1 readmission
icu_admissionid_only_1 = [k for k in p_freq if p_freq[k] ==1]

icu_admissionid_times_2 = [k for k in p_freq if p_freq[k] >=2]


icu_admission_only_1 =[]
for id in icu_admissionid_only_1:
    row = df_icu_admissions.loc[df_icu_admissions['SUBJECT_ID']==id]
    for r in row.values:
        icu_admission_only_1.append(r)
df_icu_admission_only_1 = pd.DataFrame(icu_admission_only_1,columns=df_icu_admissions.columns.values)



icu_admission_times_2 =[]
for id in icu_admissionid_times_2:
    row = df_icu_admissions.loc[df_icu_admissions['SUBJECT_ID']==id]
    for r in row.values:
        icu_admission_times_2.append(r)
df_icu_admission_times_2= pd.DataFrame(icu_admission_times_2,columns=df_icu_admissions.columns.values)



## use patient table and admission table to calculate days between readmission and age

df_patients  = pd.read_csv("PATIENTS.csv",sep=',',header=0)
df_patients.head(10)
patients = df_patients.iloc[:,1:4]
patients = pd.DataFrame(patients)

# get date pair from readmission with only 1
icuadmission_2_datepair={}
for index, data in df_icu_admission_times_2.iterrows():
    if data['SUBJECT_ID'] not in icuadmission_2_datepair:
        icuadmission_2_datepair[data['SUBJECT_ID']] =[data['INTIME']]
    else:
        icuadmission_2_datepair[data['SUBJECT_ID']].append(data['INTIME'])


# add age to date pair and output the file
icuadmission_2_datepair_age =[]
for id in icuadmission_2_datepair:
    temp = [id]
    dt1 = datetime.strptime(icuadmission_2_datepair[id][0], '%Y-%m-%d %H:%M:%S')
    dt2 = datetime.strptime(icuadmission_2_datepair[id][1], '%Y-%m-%d %H:%M:%S')
    temp.extend([dt1,dt2])
    temp.append(abs((dt2 - dt1).days))


    ageline = patients.loc[patients['SUBJECT_ID'] == id]
    dob = datetime.strptime(ageline['DOB'].item(), '%Y-%m-%d %H:%M:%S')
    temp.append(dob)

    age = dt1.year-dob.year

    if age >=100:
        temp.append(91.4)
    else:
        temp.append(age)

    icuadmission_2_datepair_age.append(temp)

df_icuadmission_2_datepair_age = pd.DataFrame(icuadmission_2_datepair_age,columns=['SUBJECT_ID','ADMITTIME','2ND_ADMITTIME','DAYS','DOB','AGE'])
df_icuadmission_2_datepair_age = df_icuadmission_2_datepair_age.sort_values('AGE')
#df_icuadmission_1_datepair_age.to_csv('3_readmission_datepair_age.csv')


# patient age distribution:
#plt.plot( [x for x in df_icuadmission_2_datepair_age['AGE']])
#plt.xlabel('Patient')
#plt.ylabel('Age')
print 'Pediatric : ', sum(df_icuadmission_2_datepair_age['AGE']==0.0)
print 'Adult : ', sum(df_icuadmission_2_datepair_age['AGE']!=0.0)




df_admissions  = pd.read_csv("ADMISSIONS.csv",sep=',',header=0)
df_admissions.head(10)



icu_diagnosis ={}
for id in icuadmission_2_datepair_age:
    row = df_admissions[df_admissions['SUBJECT_ID']==id[0]]
    diagnosis = tuple(row['DIAGNOSIS'].values[0:2])

    if diagnosis not in icu_diagnosis:
        icu_diagnosis[diagnosis] = 1
    else:
        icu_diagnosis[diagnosis] += 1

for i in icu_diagnosis:
    print i, icu_diagnosis[i]




icuadmission_2_datepair_age_30adulsts =[]
readmission_datepair_age_location_pair_ids =[]
c = 0
for pid,loc in readmission_location_pair.iteritems():
    if loc == ('EMERGENCY ROOM ADMIT', 'EMERGENCY ROOM ADMIT') or loc[1] =='EMERGENCY ROOM ADMIT' :
        row = df_readmission_datepair_age.loc[df_readmission_datepair_age['SUBJECT_ID']==pid]
        c += 1
        if row['DAYS'].values <=30 and row['AGE'].values != 0.0:
            readmission_datepair_age_location_pair.append([row.values,',',pid, ',',loc])
            readmission_datepair_age_location_pair_ids.append(pid)

df_readmission_datepair_age_location_pair = pd.DataFrame(readmission_datepair_age_location_pair)
df_readmission_datepair_age_location_pair.to_csv('df_readmission_datepair_age_location_pair.csv')









# get date pair from readmission with only 1
icuadmission_1_datepair={}
for index, data in df_icu_admission_only_1.iterrows():
    if data['SUBJECT_ID'] not in icuadmission_1_datepair:
        icuadmission_1_datepair[data['SUBJECT_ID']] =[data['INTIME']]
    else:
        icuadmission_1_datepair[data['SUBJECT_ID']].append(data['INTIME'])



icuadmission_1_datepair_age =[]
for id in icuadmission_1_datepair:
    temp = [id]
    dt1 = datetime.strptime(icuadmission_1_datepair[id][0], '%Y-%m-%d %H:%M:%S')
    temp.append(dt1)

    ageline = patients.loc[patients['SUBJECT_ID'] == id]
    dob = datetime.strptime(ageline['DOB'].item(), '%Y-%m-%d %H:%M:%S')
    temp.append(dob)

    age = dt1.year-dob.year

    if age >=100:
        temp.append(91.4)
    else:
        temp.append(age)

    icuadmission_1_datepair_age.append(temp)

df_icuadmission_1_datepair_age = pd.DataFrame(icuadmission_1_datepair_age,columns=['SUBJECT_ID','INTIME','DOB','AGE'])
df_icuadmission_1_datepair_age = df_icuadmission_1_datepair_age.sort_values('AGE')
#df_icuadmission_1_datepair_age.to_csv('3_readmission_datepair_age.csv')


# patient age distribution:
#plt.plot( [x for x in df_icuadmission_1_datepair_age['AGE']])
#plt.xlabel('Patient')
#plt.ylabel('Age')
print 'Pediatric : ', sum(df_icuadmission_1_datepair_age['AGE']==0.0)
print 'Adult : ', sum(df_icuadmission_1_datepair_age['AGE']!=0.0)



















#
# icu_13033 = df_icu_admissions[df_icu_admissions['SUBJECT_ID']==13033]
# icu =[]
# ha_icu_diff =[]
# for id in readmissionid_only_1:
#     row1 = df_admissions[df_admissions['SUBJECT_ID'] == id]
#     row2 = df_icu_admissions[df_icu_admissions['SUBJECT_ID'] == id]
#     ha_icu_diff.append([id, len(row1['SUBJECT_ID'])- len(row2['SUBJECT_ID'])])
#
# t2 = [x[1] for x in temp]
#
# t2_count ={}
# for x in t2:
#     if x in t2_count:
#         t2_count[x] += 1
#     else:
#         t2_count[x] = 1
#
# pd.DataFrame([[i,k] for i,k in t2_count.iteritems()]).to_csv('icu_test.csv')
#
# plt.plot(t2)




########### for counting subject id repeats


df_admissions  = pd.read_csv("ADMISSIONS.csv",sep=',',header=0)
df_admissions.head(10)




data = [data for index,data in df_admissions.iteritems()]
subject_id = data[1]


# get the frequency table for multiple patient id record
p_freq ={}
for x in subject_id:
    if x not in p_freq:
        p_freq[x] = 1
    else:
        p_freq[x] += 1
plt.plot(range(len(p_freq)),p_freq.values())
plt.xlabel('Patients')
plt.ylabel('Times readmitted')
plt.show()



max(p_freq, key=p_freq.get)



# get patient id with only 1 readmission
readmissionid_only_1 = [k for k in p_freq if p_freq[k] ==2]


## initial data

print "Total admission record : " , len(df_admissions)
print "Total unique record i.e total patients : " , len(p_freq)
print "Total patients who were readmitted : " , len([k for k in p_freq if p_freq[k] >1])
print "Total patients who were readmitted only once : " , len([k for k in p_freq if p_freq[k] ==2])
#print "Total number of times patients who were readmitted : " , sum([k for k in p_freq.values() if k >1])

# output the file with all patient info for 1 readmission only
readmission_1 =[]
for id in readmissionid_only_1:
    row = df_admissions.loc[df_admissions['SUBJECT_ID']==id]
    for r in row.values:
        readmission_1.append(r)
df_readmission_1 = pd.DataFrame(readmission_1,columns=df_admissions.columns.values)
df_readmission_1.to_csv('1_output_readmission_1.csv',index=False)



## this section is to calculate readmission pair group
# get pair ( 1st admission and 2nd admission - admission locations)
readmission_location_pair={}
for id in readmissionid_only_1:
    row = df_admissions.loc[df_admissions['SUBJECT_ID']==id]
    readmission_location_pair[id]=tuple(row['ADMISSION_LOCATION'].values)





# count how many for each pair
readmission_locationpair_group ={}
for item in readmission_location_pair:
    if readmission_location_pair[item] not in readmission_locationpair_group:
        readmission_locationpair_group[readmission_location_pair[item]] = 1
    else:
        readmission_locationpair_group[readmission_location_pair[item]] += 1


readmission_locationpair_group_temp =[]
for i in readmission_locationpair_group:
    readmission_locationpair_group_temp.append([i,readmission_locationpair_group[i]])

df_readmission_pair_group = pd.DataFrame(readmission_locationpair_group_temp) # , columns=['1st - 2nd Admission', 'Counts'])
df_readmission_pair_group[0] = df_readmission_pair_group[0].map(lambda x: str(x))
df_readmission_pair_group[0] = df_readmission_pair_group[0].map(lambda x: x.lstrip('(').rstrip(')'))
df_readmission_pair_group = df_readmission_pair_group.sort_values(1,ascending=True)
df_readmission_pair_group.to_csv('2_output_readmission_pair.csv',index=False)


## use patient table and admission table to calculate days between readmission and age

df_patients  = pd.read_csv("PATIENTS.csv",sep=',',header=0)
df_patients.head(10)
patients = df_patients.iloc[:,1:4]
patients = pd.DataFrame(patients)

# get date pair from readmission with only 1
readmission_datepair={}
for index, data in df_readmission_1.iterrows():
    if data['SUBJECT_ID'] not in readmission_datepair:
        readmission_datepair[data['SUBJECT_ID']] =[data['ADMITTIME']]
    else:
        readmission_datepair[data['SUBJECT_ID']].append(data['ADMITTIME'])


# add age to date pair and output the file
readmission_datepair_age =[]
for id in readmission_datepair:
    temp = [id]
    dt1 = datetime.strptime(readmission_datepair[id][0], '%Y-%m-%d %H:%M:%S')
    dt2 = datetime.strptime(readmission_datepair[id][1], '%Y-%m-%d %H:%M:%S')
    temp.extend([dt1,dt2])
    temp.append(abs((dt2 - dt1).days))


    ageline = patients.loc[patients['SUBJECT_ID'] == id]
    dob = datetime.strptime(ageline['DOB'].item(), '%Y-%m-%d %H:%M:%S')
    temp.append(dob)

    age = dt1.year-dob.year

    if age >=100:
        temp.append(91.4)
    else:
        temp.append(age)

    readmission_datepair_age.append(temp)

df_readmission_datepair_age = pd.DataFrame(readmission_datepair_age,columns=['SUBJECT_ID','ADMITTIME','2ND_ADMITTIME','DAYS','DOB','AGE'])
df_readmission_datepair_age = df_readmission_datepair_age.sort_values('AGE')
df_readmission_datepair_age.to_csv('3_readmission_datepair_age.csv')


# patient age distribution:
plt.plot( [x for x in df_readmission_datepair_age['AGE']])
plt.xlabel('Patient')
plt.ylabel('Age')
print 'Pediatric : ', sum(df_readmission_datepair_age['AGE']==0.0)
print 'Adult : ', sum(df_readmission_datepair_age['AGE']!=0.0)


### to cout patient with days group and plot distribution in each group
readmission_blocks =[7,30,60,90,180,365,4121]

readmission_days = {k:0 for k in readmission_blocks }
readmission_days = collections.OrderedDict(sorted(readmission_days.items()))

readmission_days_dist = {k:[] for k in readmission_blocks}
readmission_days_dist = collections.OrderedDict(sorted(readmission_days_dist.items()))

## now fill up the values for days and days distribution

for t in df_readmission_datepair_age['DAYS']:
    if t<=7 :
        readmission_days[7] += 1
        readmission_days_dist[7].append(t)
    elif t>7 and t<=30:
        readmission_days[30] += 1
        readmission_days_dist[30].append(t)
    elif t>30 and t<=60:
        readmission_days[60] += 1
        readmission_days_dist[60].append(t)
    elif t>60 and t<=90:
        readmission_days[90] += 1
        readmission_days_dist[90].append(t)
    elif t>90 and t<=180:
        readmission_days[180] += 1
        readmission_days_dist[180].append(t)
    elif t>180 and t<=365:
        readmission_days[365] += 1
        readmission_days_dist[365].append(t)
    elif t > 365:
        readmission_days[4121] += 1
        readmission_days_dist[4121].append(t)

 ### plot days total
df_readmission_days = pd.DataFrame([ [k,i]for k,i in readmission_days.iteritems()],columns=['DAYS','COUNTS'])
df_readmission_days = df_readmission_days.sort_values('DAYS')
y = (df_readmission_days['DAYS'])
y_pos = np.arange(len(y))
plt.bar(y_pos,df_readmission_days['COUNTS'])
plt.xticks(y_pos,y)
plt.ylabel('Patients')
plt.xlabel('Readmission Days')



## plot days  distribution

i=0
for item in readmission_days_dist:
    i += 1
    plt.subplot(1,7,i)
    plt.hist(readmission_days_dist[item],bins=10)
    plt.xlabel(str(item)+ ' Days ')
    if i ==1:
        plt.ylabel('Total Patient')
plt.show()




print readmission_location_pair
print df_readmission_datepair_age.head(2)

readmission_datepair_age_location_pair =[]
readmission_datepair_age_location_pair_ids =[]
c = 0
for pid,loc in readmission_location_pair.iteritems():
    if loc == ('EMERGENCY ROOM ADMIT', 'EMERGENCY ROOM ADMIT') or loc[1] =='EMERGENCY ROOM ADMIT' :
        row = df_readmission_datepair_age.loc[df_readmission_datepair_age['SUBJECT_ID']==pid]
        c += 1
        if row['DAYS'].values <=30 and row['AGE'].values != 0.0:
            readmission_datepair_age_location_pair.append([row.values,',',pid, ',',loc])
            readmission_datepair_age_location_pair_ids.append(pid)

df_readmission_datepair_age_location_pair = pd.DataFrame(readmission_datepair_age_location_pair)
df_readmission_datepair_age_location_pair.to_csv('df_readmission_datepair_age_location_pair.csv')


readmission_diagnosis_pair=[]

for i in readmission_datepair_age_location_pair_ids:
    row = df_readmission_1[df_readmission_1['SUBJECT_ID']==i]
    temp = [i,row['DIAGNOSIS'].values]
    readmission_diagnosis_pair.append(temp)

df_readmission_diagnosis_pair = pd.DataFrame(readmission_diagnosis_pair)
df_readmission_diagnosis_pair.to_csv('df_readmission_diagnosis_pair.csv',index=False)




row = df_readmission_1[df_readmission_1['SUBJECT_ID']==362]
row['DIAGNOSIS'].values

diagnosis_pair={}
for i,r in df_readmission_diagnosis_pair.iterrows():
    dpair = tuple(r[1])
    if dpair not in diagnosis_pair:
        diagnosis_pair[dpair] = 1
    else:
        diagnosis_pair[dpair] += 1

pd.DataFrame([x for x in diagnosis_pair.iteritems()]).to_csv('diagnosis_pair.csv',index=False)

