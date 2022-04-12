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
import json


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
show = diabetic_data['medical_specialty'].value_counts()

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

# =============================================================================
# Splitting the erroneously concatenated dataframes 
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
# Encoding specific column values 
# NOTE: can manually induce new columns and then splicing new dataframes with only the columns of interest; alternatively use for-loop 
# =============================================================================
len(diabetic_data[(diabetic_data == 'nan').any(axis=1)])
len(diabetic_data[~(diabetic_data == 'nan').any(axis=1)])

diabeticSamp = diabetic_data.sample(50)
diabeticSamp[['age']].dtypes
list(diabeticSamp.age)
diabeticSamp[['race']].dtypes 

############## Testing with sample ##############
samp_cols = diabeticSamp.columns
samp_cols_obj = diabeticSamp.select_dtypes(['object']).columns
#cols_obj += '_encoded'
samp_to_cat = list(diabeticSamp.select_dtypes(['object']).columns)

test_list = []

for name in samp_to_cat:
    diabeticSamp[name] = diabeticSamp[name].astype('category')
    d = dict(enumerate(diabeticSamp[name].cat.categories))
    test_list.append(d)
    
diabeticSamp[samp_cols_obj] = diabeticSamp[samp_cols_obj].apply(lambda x: x.cat.codes)

'''entry =['a', 'b', 'c']
case_list = []
for entry in entries_list:
    case = {'key1': entry[0], 'key2': entry[1], 'key3':entry[2] }
    case_list.append(case)'''
    
# =============================================================================
##################### Reverse Mapping is WORK IN PROGRESS #####################
# =============================================================================

'''from collections import defaultdict
converted = defaultdict(list)

for attribute, code in test_list:
    converted['attribute'].append(code)

sorted(converted.items())


diabeticSamp['race_reversed'] = diabeticSamp['race'].map(nested_d)'''


############## Initialize to main dataframe ##############
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
# Exporting files 
# =============================================================================

path = os.getcwd()
cleaned_diabetic_data.to_csv(path + '/Desktop/cleaned_diabetic_data.csv', encoding = 'utf-8' )
test = pd.read_csv('/Users/William/Desktop/cleaned_diabetic_data.csv')

file = path + '/Desktop/cleaned_diabetic_data_dict.txt'    
with open(file, 'w') as f:
    for item in main_dictionary:
        f.write("%s\n" % item)

file2 = path + '/Desktop/mapping_dictionary.json'
with open(file2, 'w') as f:
     f.write(json.dumps(mapping_dictionary))

# =============================================================================
# Filtering out new dataframes
# =============================================================================
