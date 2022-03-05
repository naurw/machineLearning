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
diabetic_data.replace('?', 'NaN', inplace = True)
# May consider grouping NaNs to Other for statistical analyses selectively / per columns 

# =============================================================================
# Adding space in between potential words using loop + join() 
# NOTE: using import re is much simpler but for experimental purposes, this method was chosen 
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
    re.sub(r'\W+', '')
    return
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
# =============================================================================
test = diabetic_data['medical_specialty'].to_list()
res2 = [re.sub(r"(\w)([A-Z])", r"\1 \2", ele) for ele in test]
diabetic_data['medical_specialty'] = res2

show = diabetic_data['medical_specialty'].value_counts()
