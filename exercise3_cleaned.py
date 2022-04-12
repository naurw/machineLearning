#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  2 13:26:08 2022

@author: William
"""

import pandas as pd 
import numpy as np 
import os 
import glob 
import re 
import matplotlib.pyplot as plt 
import seaborn as sns 
import missingno as msno 
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
import plotly.express as px
import plotly.graph_objs as go
import plotly.offline as py
from plotly.offline import plot
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
import warnings
warnings.filterwarnings("ignore")

path = os.getcwd()
csv_files = glob.glob(os.path.join(path, 'Desktop/SBU/HHA 550', '*.csv'))
csv_files

csvList = []
for i in csv_files: 
    temp = pd.read_csv(i)
    csvList.append(temp)
    print('File Name:', i)
    print(temp)
    
overview = pd.concat(csvList)
overviewSamp = overview.sample(25)
diabetic_data = pd.DataFrame(csvList[1])
id_mapping = pd.DataFrame(csvList[0])

len(diabetic_data[(diabetic_data == '?').any(axis=1)])
len(diabetic_data[~(diabetic_data == '?').any(axis=1)])
diabetic_data.race.value_counts()
diabetic_data.race.replace('?', 'Other', inplace = True)
diabetic_data.gender.replace('Unknown/Invalid', 'Other', inplace = True)
diabetic_data.replace('?', 'NaN', inplace = True)
# =============================================================================
# Splitting the erroneously concatenated dataframes within id_mapping
# NOTE: transposing is not required at all--simply for visualization purposes 
# =============================================================================
id_mapping.columns
id_mapping['groupNum'] = id_mapping.isnull().all(axis=1).cumsum()

id_mapping_dict = {n: id_mapping.iloc[rows] 
     for n, rows in id_mapping.groupby('groupNum').groups.items()}

id_mapping_dict 
print(list(id_mapping_dict))

admission_type_id = id_mapping_dict[0].drop(columns= ['groupNum']).dropna(how='all')
discharge_disposition_id = id_mapping_dict[1].drop([8,9]).reset_index(drop=True)
discharge_disposition_id.drop(columns= ['groupNum'], inplace=True)
discharge_disposition_id.rename(columns ={'admission_type_id':'discharge_disposition_id'}, inplace= True)
admission_source_id = id_mapping_dict[2].drop([40,41]).reset_index(drop=True)
admission_source_id.drop(columns= ['groupNum'], inplace=True)
admission_source_id.rename(columns ={'admission_type_id':'admission_source_id'}, inplace= True)
admission_source_id['description'] = admission_source_id['description'].str.strip()

# =============================================================================
# Reformatting/cleaning data 
# =============================================================================
diabetic_data['race']
test1 = diabetic_data['race'].to_list()

print('the original list:\n', test1)
#print('the original list:\n' + str(test1))

res = []

for ele in test1: 
    temp = [[]]
    for char in ele: 
        if char.isupper():
            temp.append([])
            
        temp[-1].append(char)
        
    res.append(' '.join(''.join(ele) for ele in temp).strip())
 
print('the modified list:\n' + str(res))

diabetic_data['race'] = res
diabetic_data['race'].value_counts()

# =============================================================================
# Removing special characters in between words using regex + map() + lambda 
# NOTE: for loop + join() would work but is overly complicated and inefficient to write out 
# =============================================================================
diabetic_data['medical_specialty'] = diabetic_data['medical_specialty'].map(lambda x: re.sub(r'\W+', '', x))
diabetic_data['medical_specialty'].value_counts()

# =============================================================================
# Adding space in between potential words using regex() + list comprehension 
# NOTE: regex formula will remove 
# =============================================================================
test = diabetic_data['medical_specialty'].to_list()
res2 = [re.sub(r"(\w)([A-Z][a-z])", r"\1 \2", ele) for ele in test]
diabetic_data['medical_specialty'] = res2

diabetic_data['medical_specialty'].replace('Physician Not Found', 'nan', inplace = True)

show = diabetic_data['medical_specialty'].value_counts() # <-- manual cleaning is required for removing nested words e.g.: and, within 

diabetic_data.loc[diabetic_data['medical_specialty'] == 'Surgery Plasticwithin Headand Neck', 'medical_specialty'] = 'Surgery Plastic within Head and Neck'
diabetic_data.loc[diabetic_data['medical_specialty'] == 'Pediatrics Allergyand Immunology', 'medical_specialty'] = 'Pediatrics Allergy and Immunology'
diabetic_data.loc[diabetic_data['medical_specialty'] == 'Physical Medicineand Rehabilitation', 'medical_specialty'] = 'Physical Medicine and Rehabilitation'
diabetic_data.loc[diabetic_data['medical_specialty'] == 'Obstetricsand Gynecology', 'medical_specialty'] = 'Obstetrics and Gynecology'
diabetic_data.loc[diabetic_data['medical_specialty'] == 'Allergyand Immunology', 'medical_specialty'] = 'Allergy and Immunology'
diabetic_data.loc[diabetic_data['medical_specialty'] == 'Obsterics Gynecology Gynecologic Onco', 'medical_specialty'] = 'Obstetrics Gynecology Gynecologic Onco'

len(diabetic_data[(diabetic_data == 'NaN').any(axis=1)])
len(diabetic_data[~(diabetic_data == 'NaN').any(axis=1)])
diabetic_data.replace('NaN', np.nan, inplace = True) #replace previous 'nan' with np.nan for missingno 
# =============================================================================
# Graphs for initial review 
# =============================================================================
def missing (i):
    missing_number = i.isnull().sum().sort_values(ascending=False).to_frame()
    missing_percent = (i.isnull().sum()/i.isnull().count()).sort_values(ascending=False).round(3).to_frame()
    missing_values = pd.concat([missing_number, missing_percent], axis=1, keys=['Missing_Number', 'Missing_Percent'])
    missing_values['Missing_Percent'] = missing_values["Missing_Percent"]*100
    return missing_values

#Visualizing the distribution of null values
missing_values = missing(diabetic_data)

missing_values_graph = msno.bar(diabetic_data)
missing_values_matrix = msno.matrix(diabetic_data)

y = diabetic_data['readmitted']
y.value_counts()
y.value_counts(normalize=True)
y.value_counts(normalize=True)[0]
y.value_counts(normalize=True)[1]
y.value_counts(normalize=True)[2]
print(f'Percentage of patient(s) had been readmitted for <30 days: {round(y.value_counts(normalize=True)[1]*100,2)}% --> ({y.value_counts()[1]} patients)\nPercentage of patient(s) had not been readmitted: {round(y.value_counts(normalize=True)[0]*100,2)}% --> ({y.value_counts()[0]} patients)\nPercentage of patient(s) had been readmitted for >30 days: {round(y.value_counts(normalize=True)[2]*100,2)}% --> ({y.value_counts()[2]} patient)')


diabetic_data.info()
num = list(diabetic_data.select_dtypes(['int64']).columns)
num_exclude = ['encounter_id', 'patient_nbr', 'admission_type_id', 'discharge_disposition_id', 'admission_source_id']
num_cleaned = [x for x in num if x not in num_exclude]
cat = list(diabetic_data.select_dtypes(['object']).columns)
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

stats = diabetic_data[num].describe()
stats.info()
stats.drop(columns = ['encounter_id', 'patient_nbr', 'admission_type_id', 'discharge_disposition_id', 'admission_source_id'], inplace = True)
stats1 = diabetic_data[num_cleaned].skew()
stats2 = diabetic_data[num_cleaned]

diabetic_data[num_cleaned].hist(figsize=(20,10));

fig = px.histogram(diabetic_data, x="readmitted", title='Readmission', width=400, height=400)
plot(fig)

fig = px.histogram(diabetic_data, x="race", title='Race', width=500, height=500)
plot(fig)

fig = px.histogram(diabetic_data, x="gender", title='Gender', width=500, height=500)
plot(fig)

diabetic_data.gender.value_counts()

fig = px.histogram(diabetic_data, x="medical_specialty", title='Medical Specialty', width=500, height=500)
plot(fig)

fig = px.histogram(diabetic_data, x="age", title='Age', width=500, height=500)
plot(fig)

fig = px.histogram(diabetic_data, x="age", color="readmitted", title = 'Age vs Readmission',width=600, height=600)
plot(fig)

fig = px.histogram(diabetic_data, x="race", color="readmitted", title = 'Race vs Readmission',width=600, height=600)
plot(fig)

fig = px.histogram(diabetic_data, x="gender", color="readmitted", title = 'Gender vs Readmission',width=600, height=600)
plot(fig)

fig = px.histogram(diabetic_data, x="medical_specialty", color="readmitted", title = 'Medical Specialty vs Readmission',width=600, height=600)
plot(fig)

fig = px.scatter(diabetic_data, x='age', y='race', title='Age and Race ',color='readmitted', hover_data = diabetic_data[['readmitted']])
plot(fig)

diabetic_data.duplicated().sum()

stats3 = diabetic_data[num_cleaned].corr()
stats3_mean = diabetic_data[num_cleaned].mean()
stats3_grouped_mean = diabetic_data.groupby('readmitted')[num_cleaned].mean()

# =============================================================================
# Dropping irrelevant columns 
# =============================================================================
diabetic_data.info()
diabetic_data.drop(columns = {'weight', 'payer_code', 'medical_specialty'}, inplace= True)
diabetic_data.isnull().sum()
msno.bar(diabetic_data)

# =============================================================================
# Data dictionary + encoding 
# =============================================================================
cleaned_diabetic_data = diabetic_data.copy(deep = True) 
cols = cleaned_diabetic_data.columns
cols_obj = cleaned_diabetic_data.select_dtypes(['object']).columns
to_cat = list(cleaned_diabetic_data.select_dtypes(['object']).columns)

main_dictionary = []

for name in to_cat:
    cleaned_diabetic_data[name] = cleaned_diabetic_data[name].astype('category')
    d = dict(enumerate(cleaned_diabetic_data[name].cat.categories))
    main_dictionary.append(d)
    
cleaned_diabetic_data[cols_obj] = cleaned_diabetic_data[cols_obj].apply(lambda x: x.cat.codes)

mapping_dictionary = dict(zip(to_cat, main_dictionary))
mapping_dataframe = pd.DataFrame.from_dict(mapping_dictionary)

# =============================================================================
# Exporting 
# =============================================================================
cleaned_diabetic_data.to_csv(path + '/Desktop/cleaned_diabetic_data.csv', encoding = 'utf-8' )
test = pd.read_csv('/Users/William/Desktop/cleaned_diabetic_data.csv')

file = path + '/Desktop/cleaned_diabetic_data_dict.txt'    
with open(file, 'w') as f:
    for item in main_dictionary:
        f.write("%s\n" % item)

file2 = path + '/Desktop/mapping_dictionary.json'
with open(file2, 'w') as f:
     f.write(json.dumps(mapping_dictionary))
     
#jupytext --to notebook exercise3_cleaned.py
#ipynb-py-convert exercise3_cleaned.py test_clone.ipynb
#import IPython.nbformat.current as nbf
#nb = nbf.read(open('exercise3_cleaned.py', 'r'), 'py')
#nbf.write(nb, open('test.ipynb', 'w'), 'ipynb')


from sklearn.metrics import mutual_info_score
def cat_mut_inf(series):
    return mutual_info_score(series, diabetic_data['readmitted']) 

diabetic_cat = cleaned_diabetic_data[cat_cleaned].apply(cat_mut_inf) 
diabetic_cat = diabetic_cat.sort_values(ascending=False).to_frame(name='mutual_info_score') 
diabetic_cat