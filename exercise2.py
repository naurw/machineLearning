#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 19:12:21 2022

@author: William
"""

import pandas as pd 
import os 
import glob 
import re


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

diabetic_data.replace('?', 'nan', inplace = True)
# May consider grouping NaNs to Other for statistical analyses selectively / per columns 

# =============================================================================
# Adding space in between potential words using loop + join() 
# NOTE: using regex() is much simpler as you can see below, but for practice, this method was chosen 
#
# Logic: 
# 1. initialize list
# 2. loop to iterate all strings within list 
# 3. condition for checking for uppercase characters 
# 4. appending character with latest list / previous location 
# 5. joining lists after adding space 
# 6. remove excess leading and trailing white spaces post appending
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
#
#
# Logic: 
# my_function = lambda x : x.my_method(3) == def my_function(x): return x.my_method(3)
# 
# squares = lambda x: x*x == def squares_def(x): return x*x
# print('Using lambda: ', squares(5)) == print('Using def: ', squares_def(5))
# =============================================================================
show = diabetic_data['medical_specialty'].value_counts()

diabetic_data['medical_specialty'] = diabetic_data['medical_specialty'].map(lambda x: re.sub(r'\W+', '', x))
diabetic_data['medical_specialty'].value_counts()

# =============================================================================
# Adding space in between potential words using regex() + list comprehension 
# NOTE: regex formula will remove 
#
# Logic: 
# 1. initialize list 
# 2. regex() 
#
# Manual cleaning for nested words e.g.: 'and', 'within'
#
# Methods of choice (tested): 
# 1. df.loc[(df.test == '55'),'Score']='fail'
# 2. df['test'] = np.where((df.test == '100'),'perfect',df.Event)
# 3. df['test'].mask(df['test'] == '65', 'pass', inplace=True)
# 4. m = df.test == '0'
#    df.where(~m,other='fail; office hours required')
# =============================================================================
test = diabetic_data['medical_specialty'].to_list()
res2 = [re.sub(r"(\w)([A-Z][a-z])", r"\1 \2", ele) for ele in test]
diabetic_data['medical_specialty'] = res2

diabetic_data['medical_specialty'].replace('Physician Not Found', 'nan', inplace = True)

show = diabetic_data['medical_specialty'].value_counts() # <-- manual cleaning is required for removing nested words e.g.: and, within 

diabetic_data.loc[diabetic_data['medical_specialty'] == 'Surgery Plasticwithin Headand Neck', 'medical_specialty'] = 'Surgery Plastic within Head and Neck'
diabetic_data.loc[diabetic_data['medical_specialty'] == 'Pediatrics Allergyand Immunology', 'medical_specialty'] = 'Pediatrics Allergy and Immunology'
diabetic_data.loc[diabetic_data['medical_specialty'] == 'Physical Medicineand Rehabilitation', 'medical_specialty'] = 'Physical Medicine and Rehabilitation'

# =============================================================================
# Splitting the erroneously concatenated dataframes 
# NOTE: transposing is not required at all--simply for visualization purposes 
# 
# Logic: 
# 1. identify all columns with nan values and transpose
# 2. split concatenated dataframe by cumsum() values 
# 3. open each dataframe from dictionary key 
# 4. rename and format each dataframe 
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
# Encoding specific column values 
# NOTE: can manually induce new columns and then splicing new dataframes with only the columns of interest; alternatively use for-loop 
# diabetic_data['race'] = diabetic_data['race'].astype('category')
# diabetic_data[['race']].dtypes
# diabetic_data['race_encoded'] = diabetic_data['race'].cat.codes
# diabetic_data['race_encoded'].nunique()

# diabetic_data['age_encoded'] = diabetic_data['age'].astype('category').cat.codes 
# diabetic_data[['age']].dtypes
# diabetic_data['age_encoded'].nunique()
#
# Logic: 
# 1. dtype change for cat.codes 
# 2. initialize list of all object columns 
# 3. 
# =============================================================================
len(diabetic_data[(diabetic_data == 'nan').any(axis=1)])
len(diabetic_data[~(diabetic_data == 'nan').any(axis=1)])

diabeticSamp = diabetic_data.sample(50)
diabeticSamp[['age']].dtypes
list(diabeticSamp.age)
diabeticSamp[['race']].dtypes 

# Testing with sample 
cols = diabeticSamp.columns
cols_obj = diabeticSamp.select_dtypes(['object']).columns
to_cat = list(diabeticSamp.select_dtypes(['object']).columns)

for name in to_cat:
    diabeticSamp[name] = diabeticSamp[name].astype('category')
    
diabeticSamp[cols_obj] = diabeticSamp[cols_obj].apply(lambda x: x.cat.codes)

# Initialize to main dataframe 
diabetic_data.race.dtypes
cols = diabetic_data.columns
cols_obj = diabetic_data.select_dtypes(['object']).columns
to_cat = list(diabetic_data.select_dtypes(['object']).columns)

for name in to_cat:
    diabetic_data[name] = diabetic_data[name].astype('category')
    
diabetic_data[cols_obj] = diabetic_data[cols_obj].apply(lambda x: x.cat.codes)


# =============================================================================
# Filtering out new dataframes
# =============================================================================
