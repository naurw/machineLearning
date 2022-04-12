#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 17:35:06 2022

@author: William
"""

import pandas as pd 
import numpy as np 
import os 
import glob 
import re 
import missingno as msno 
import random
import plotly.express as px
import plotly.offline as py
from plotly.offline import plot
import warnings 
warnings.filterwarnings("ignore")


import matplotlib.pyplot as plt 
import seaborn as sns 
import csv 
import json 
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, classification_report, make_scorer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.model_selection import KFold,cross_val_score, RepeatedStratifiedKFold,StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import OneHotEncoder,StandardScaler,PowerTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import KNNImputer,SimpleImputer
from sklearn.compose import make_column_transformer
from imblearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.impute import SimpleImputer
from sklearn.dummy import DummyClassifier
from imblearn.over_sampling import SMOTE
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, balanced_accuracy_score,\
                            precision_score, recall_score, roc_auc_score,\
                            plot_confusion_matrix, classification_report, plot_roc_curve, f1_score
import plotly 
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff


# Loading in files 
path = os.getcwd()
csv_files = glob.glob(os.path.join(path, 'Desktop/SBU/HHA 550', '*.csv'))
csv_files

csvList = []
for i in csv_files: 
    temp = pd.read_csv(i)
    csvList.append(temp)
    print('File Name:', i)
    print(temp)
    
    for temp in csvList: 
        overview = pd.concat(csvList)

id_mapping = pd.DataFrame(csvList[0])
diabetic_data = pd.DataFrame(csvList[1])

# Splitting concatenated dataframes into separate dataframes         
id_mapping.columns
id_mapping['groupNum'] = id_mapping.isnull().all(axis=1).cumsum()

id_mapping_dict = {n: id_mapping.iloc[rows] for n, rows in id_mapping.groupby('groupNum').groups.items()}
print(id_mapping_dict)

admission_type_id = id_mapping_dict[0].drop(columns= ['groupNum']).dropna(how='all')
discharge_disposition_id = id_mapping_dict[1].drop([8,9]).reset_index(drop=True)
discharge_disposition_id.drop(columns= ['groupNum'], inplace=True)
discharge_disposition_id.rename(columns ={'admission_type_id':'discharge_disposition_id'}, inplace= True)
admission_source_id = id_mapping_dict[2].drop([40,41]).reset_index(drop=True)
admission_source_id.drop(columns= ['groupNum'], inplace=True)
admission_source_id.rename(columns ={'admission_type_id':'admission_source_id'}, inplace= True)
admission_source_id['description'] = admission_source_id['description'].str.strip()

# Dealing with ?s within diabetic_data 
def nans (i):
    i.race.replace('?', 'Other', inplace = True)
    i.gender.replace('Unknown/Invalid', 'Other', inplace = True)
    i.medical_specialty.replace('Physician Not Found', 'NaN', inplace = True)
    i.replace('?', 'NaN', inplace = True)
    print('The total number of NaNs within this dataset is ', len(i[(i == 'NaN').any(axis=1)]), '.', sep='')
    print('The total number of rows withibout NaNs within this dataset is ', len(i[~(i == 'NaN').any(axis=1)]), '.', sep='')

nans(diabetic_data)

# Formating the strings within certain columns 
t = diabetic_data['race'].to_list()
print('the original list:\n', random.sample(t, 50))
     #print('the original list:\n' + str(t))

res = []
for ele in t: 
    temp = [[]]
    
    for char in ele: 
        if char.isupper():
            temp.append([])
            
        temp[-1].append(char)
        
    res.append(' '.join(''.join(ele) for ele in temp).strip())
diabetic_data.race = res
 
#print('the modified list:\n' + str(res))
diabetic_data['race'].value_counts()

def ele (i): 
    i.medical_specialty = i.medical_specialty.map(lambda x: re.sub(r'\W+', '', x))
    t2 = i.medical_specialty.to_list()
    res2 = [re.sub(r"(\w)([A-Z][a-z])", r"\1 \2", i) for i in t2]
    i.medical_specialty = res2 
    
ele(diabetic_data)
t3 = diabetic_data['medical_specialty'].value_counts()    

diabetic_data.loc[diabetic_data['medical_specialty'] == 'Surgery Plasticwithin Headand Neck', 'medical_specialty'] = 'Surgery Plastic within Head and Neck'
diabetic_data.loc[diabetic_data['medical_specialty'] == 'Pediatrics Allergyand Immunology', 'medical_specialty'] = 'Pediatrics Allergy and Immunology'
diabetic_data.loc[diabetic_data['medical_specialty'] == 'Physical Medicineand Rehabilitation', 'medical_specialty'] = 'Physical Medicine and Rehabilitation'
diabetic_data.loc[diabetic_data['medical_specialty'] == 'Obstetricsand Gynecology', 'medical_specialty'] = 'Obstetrics and Gynecology'
diabetic_data.loc[diabetic_data['medical_specialty'] == 'Allergyand Immunology', 'medical_specialty'] = 'Allergy and Immunology'
diabetic_data.loc[diabetic_data['medical_specialty'] == 'Obsterics Gynecology Gynecologic Onco', 'medical_specialty'] = 'Obstetrics Gynecology Gynecologic Onco'

# Visualizing and find the distributuion of the data including the missing values 
def missing (i):
    i.replace('NaN', np.nan, inplace = True) #replace previous 'nan' with np.nan for coutning missing values  
    missing_number = i.isnull().sum().sort_values(ascending=False).to_frame()
    missing_percent = (i.isnull().sum()/i.isnull().count()).sort_values(ascending=False).round(3).to_frame()
    missing_values = pd.concat([missing_number, missing_percent], axis=1, keys=['Missing_Number', 'Missing_Percent'])
    missing_values['Missing_Percent'] = missing_values["Missing_Percent"]*100
    return missing_values

missing_values = missing(diabetic_data)
msno.bar(diabetic_data)
msno.matrix(diabetic_data)


# Nonbinary variable of interest --> chnage <30 and >30 to YES 
diabetic_data.readmitted.value_counts()
binary_diabetic = diabetic_data.copy(deep = True)
yes = ['>30', '<30']
binary_diabetic.loc[binary_diabetic['readmitted'].isin(yes), 'readmitted'] = 'YES'

binary_diabetic.info()
num = list(binary_diabetic.select_dtypes(['int64']).columns)
num_exclude = ['encounter_id', 'patient_nbr', 'admission_type_id', 'discharge_disposition_id', 'admission_source_id']
num_cleaned = [x for x in num if x not in num_exclude]
cat = list(binary_diabetic.select_dtypes(['object']).columns)
cat_exclude = ['max_glu_serum',
 'A1Cresult',
 'metformin',
 'repaglinide',
 'nateglinide',
 'chlorpropamide',
 'glimepiride',
 'acetohexamide',
 'glipizide',
 'glyburide',
 'tolbutamide',
 'pioglitazone',
 'rosiglitazone',
 'acarbose',
 'miglitol',
 'troglitazone',
 'tolazamide',
 'examide',
 'citoglipton',
 'insulin',
 'glyburide-metformin',
 'glipizide-metformin',
 'glimepiride-pioglitazone',
 'metformin-rosiglitazone',
 'metformin-pioglitazone',
 'change']
cat_cleaned = [x for x in cat if x not in cat_exclude]
cat_cleaned 
'''['race',
 'gender',
 'age',
 'weight',
 'payer_code',
 'medical_specialty',
 'diag_1',
 'diag_2',
 'diag_3',
 'diabetesMed',
 'readmitted']'''

# Interpreting numerical features in relation to readmission 
# =============================================================================
# mean < median = left skew (negative) 
# mean > median = right skew (positive)
# skew = lopsidedness | frequency 
# kurt = tailedness | peakness 
# =============================================================================
stats = binary_diabetic[num_cleaned].describe()
stats
stats1 = binary_diabetic[num_cleaned].skew()
stats1
stats2 = binary_diabetic[num_cleaned].corr()
stats2
stats3 = binary_diabetic[num_cleaned].kurtosis()
stats3
stats4 = binary_diabetic[num_cleaned].mean()
stats4
stats4_grouped = binary_diabetic.groupby('readmitted')[num_cleaned].mean()
stats4_grouped 


binary_diabetic[num_cleaned].hist(figsize=(20,10));

# Normalize: finding the rate of occurences rather than the count of occurences 
y = diabetic_data['readmitted']
y.value_counts()
y.value_counts(normalize=True)
#y.value_counts(normalize=True)[0]
#y.value_counts(normalize=True)[1]
#y.value_counts(normalize=True)[2]
print(f'Percentage of patient(s) had been readmitted for <30 days: {round(y.value_counts(normalize=True)[1]*100,2)}% --> ({y.value_counts()[1]} patients)\nPercentage of patient(s) had not been readmitted: {round(y.value_counts(normalize=True)[0]*100,2)}% --> ({y.value_counts()[0]} patients)\nPercentage of patient(s) had been readmitted for >30 days: {round(y.value_counts(normalize=True)[2]*100,2)}% --> ({y.value_counts()[2]} patient)')

# =============================================================================
# binary_diabetic[binary_diabetic['gender'] == 'Male' ]['readmitted'].count()
# binary_diabetic[binary_diabetic['gender'] == 'Female']['readmitted'].count()
# binary_diabetic[binary_diabetic['gender'] == 'Other']['readmitted'].count()
# binary_diabetic.gender.value_counts()
# binary_diabetic['gender'].value_counts(normalize = True)
# =============================================================================

print (f'{round(binary_diabetic["race"].value_counts(normalize=True)*100,2)}')
print (f'{round(binary_diabetic["gender"].value_counts(normalize=True)*100,2)}')
print (f'{round(binary_diabetic["age"].value_counts(normalize=True)*100,2)}')
print (f'{round(binary_diabetic["diabetesMed"].value_counts(normalize=True)*100,2)}')
print (f'{round(binary_diabetic["medical_specialty"].value_counts(normalize=True)*100,2).head(5)}')
print (f'{round(binary_diabetic["diabetesMed"].value_counts(normalize=True)*100,2)}')
print (f'{round(binary_diabetic["diag_1"].value_counts(normalize=True)*100,2).head(10)}')
print (f'{round(binary_diabetic["diag_2"].value_counts(normalize=True)*100,2)}')
print (f'{round(binary_diabetic["diag_3"].value_counts(normalize=True)*100,2)}')

# Focal categorical columns will be race, gender, age, diabeticsMed; other columns seem to be irrelevcant 
binary_diabetic.info()
binary_diabetic.drop(columns = {'weight', 'payer_code', 'medical_specialty', 'diag_1', 'diag_2', 'diag_3'}, inplace= True)
binary_diabetic.isnull().sum()
msno.bar(binary_diabetic)


cols = binary_diabetic.columns
cols_obj = binary_diabetic.select_dtypes(['object']).columns
to_cat = list(binary_diabetic.select_dtypes(['object']).columns)

main_dictionary = []

for name in to_cat:
    binary_diabetic[name] = binary_diabetic[name].astype('category')
    d = dict(enumerate(binary_diabetic[name].cat.categories))
    main_dictionary.append(d)
    
binary_diabetic[cols_obj] = binary_diabetic[cols_obj].apply(lambda x: x.cat.codes)

mapping_dictionary = dict(zip(to_cat, main_dictionary))
mapping_dataframe = pd.DataFrame.from_dict(mapping_dictionary)


from sklearn.metrics import mutual_info_score
def cat_mut_inf(series):
    return mutual_info_score(series, binary_diabetic['readmitted']) 

cat_cleaned_v2 = list(binary_diabetic.select_dtypes(['object']).columns)
cat_exclude_v2 = ['readmitted']
cat_cleaned_v2 = [x for x in cat_cleaned_v2 if x not in cat_exclude_v2]

binary_cat = binary_diabetic[cat_cleaned_v2].apply(cat_mut_inf).sort_values(ascending=False).to_frame(name='mutual_info_score') 
binary_cat





train_df, valid_df, test_df = np.split(binary_diabetic.sample(frac=1, random_state=42), 
                                       [int(.7*len(binary_diabetic)), int(0.85*len(binary_diabetic))])
train_df = train_df.reset_index(drop = True)
valid_df = valid_df.reset_index(drop = True)
test_df = test_df.reset_index(drop = True)

binary_diabetic.readmitted.value_counts()
train_df.readmitted.value_counts()
valid_df.readmitted.value_counts()
test_df.readmitted.value_counts()


def calc_prevalence(y_actual):
    
    '''
    This function is to understand the ratio/distribution of the classes that we are going to predict for.
    
    Params:
    1. y_actual: The target feature
    
    Return:
    1. (sum(y_actual)/len(y_actual)): The ratio of the postive class in the comlpete data.
    '''
    
    return (sum(y_actual)/len(y_actual))


rows_pos = train_df.readmitted == 1
df_train_pos = train_df.loc[rows_pos]
df_train_neg = train_df.loc[~rows_pos]


# merge the balanced data
binary_df_balanced = pd.concat([df_train_pos, df_train_neg.sample(n = len(df_train_pos), random_state = 111)],axis = 0)

# shuffle the order of training samples 
binary_df_balanced = binary_df_balanced.sample(n = len(binary_df_balanced), random_state = 42).reset_index(drop = True)

print('Train balanced prevalence(n = %d):%.3f'%(len(binary_df_balanced), \
                                                calc_prevalence(binary_df_balanced.readmitted.values)))

binary_df_balanced.stroke.value_counts()

X_train = binary_df_balanced.drop('readmitted',axis=1)

y_train = binary_df_balanced['readmitted']

X_valid = valid_df.drop('readmitted',axis=1)

y_valid = valid_df['readmitted']

X_test = test_df.drop('readmitted',axis=1)

y_test = test_df['readmitted']

scaler=StandardScaler()
X_train[['race', 'gender', 'age']] = pd.DataFrame(scaler.fit_transform(X_train[['race', 'gender', 'age']]),columns=['race', 'gender', 'age'])
X_valid[['race', 'gender', 'age']] = pd.DataFrame(scaler.transform(X_valid[['race', 'gender', 'age']]),columns=['race', 'gender', 'age'])
X_test[['race', 'gender', 'age']] = pd.DataFrame(scaler.transform(X_test[['race', 'gender', 'age']]),columns=['race', 'gender', 'age'])

"""# Creating and Understanding Models"""

def calc_specificity(y_actual, y_pred, thresh):
    # calculates specificity
    return sum((y_pred < thresh) & (y_actual == 0)) /sum(y_actual ==0)

def print_report(y_actual, y_pred, thresh = 0.5):
    
    '''
    This function calculates all the metrics to asses the machine learning models.
    
    Params:
    1. y_actual: The actual values for the target variable.
    2. y_pred: The predicted values for the target variable.
    3. thresh: The threshold for the probability to be considered as a positive class. Default value 0.5
    
    Return:
    1. AUC
    2. Accuracy
    3. Recall
    4. Precision
    5. Specificity
    '''
    
    auc = roc_auc_score(y_actual, y_pred)
    accuracy = accuracy_score(y_actual, (y_pred > thresh))
    recall = recall_score(y_actual, (y_pred > thresh))
    precision = precision_score(y_actual, (y_pred > thresh))
    specificity = calc_specificity(y_actual, y_pred, thresh)
    print('AUC:%.3f'%auc)
    print('accuracy:%.3f'%accuracy)
    print('recall:%.3f'%recall)
    print('precision:%.3f'%precision)
    print('specificity:%.3f'%specificity)
    print('prevalence:%.3f'%calc_prevalence(y_actual))
    print(' ')
    return auc, accuracy, recall, precision, specificity

"""## Linear Regression"""

lnr = LinearRegression()
lnr.fit(X_train, y_train)

y_valid_preds = lnr.predict(X_valid)
print(y_valid_preds)
'''[0.58308616 0.68517083 0.73455565 ... 0.42218064 0.75081358 0.43347143]'''

"""## Logistic Regression"""

lr=LogisticRegression(random_state = 42, solver = 'newton-cg', max_iter = 200)
lr.fit(X_train, y_train)

y_valid_preds = lr.predict_proba(X_valid)[:,1]
print(y_valid_preds)
'''[0.57723503 0.74442624 0.75745062 ... 0.40768596 0.86378158 0.41540911]'''

print('Metrics for Validation data:')
lr_valid_auc, lr_valid_accuracy, lr_valid_recall, \
    lr_valid_precision, lr_valid_specificity = print_report(y_valid,y_valid_preds, 0.5)
    '''Metrics for Validation data:
AUC:0.673
accuracy:0.630
recall:0.562
precision:0.604
specificity:0.688
prevalence:0.459'''
# =============================================================================
# Graphs and Charts 
# =============================================================================

fig = px.scatter(binary_diabetic, x='age', y='race', title='Age and Race ',color='readmitted', hover_data = binary_diabetic[['readmitted']])
plot(fig)

fig = px.scatter(binary_diabetic, x='age', y='gender', title='Age and Gender',color='readmitted', hover_data = binary_diabetic[['readmitted']])
plot(fig)

fig = px.scatter(binary_diabetic, x='race', y='gender', title='Gender and Race ',color='readmitted', hover_data = binary_diabetic[['readmitted']])
plot(fig)

fig = px.scatter(binary_diabetic, x='time_in_hospital', y='race', title='Age and Race ',color='readmitted', hover_data = binary_diabetic[['readmitted']])
plot(fig)