# -- coding: utf-8 --
"""
Created on Thu Sep 30 09:49:38 2021

@author: Usuario
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime as dt 


df=pd.read_csv('garments_worker_productivity.csv')
df.shape
df.info()
df.isnull().sum()
df.info()

#convert data column from string to datetime 
df['date']=pd.to_datetime(df['date'])
df.info()

#Solve writing problems of the departments names
df['department'].describe(include='all')
df['department'].value_counts()
df['department']=df['department'].replace(['finishing '],'finishing')
df['department'].value_counts()

#Replace 506 values missing in the 'wip' attribute with the wip mean
df['wip'].value_counts()
wip_mean=int(df['wip'].mean())
df['wip']=df['wip'].replace(np.nan,wip_mean)

#check if there still are missing values in the dataset
df.isnull().sum()

#------------------------covert to standard format------------------
raw_data=df.values
cols=range(1,14)
X=raw_data[:,cols]
attributeNames=np.asarray(df.columns[cols])

#column quarter
quarterLabels=raw_data[:,1]
quarterNames=np.unique(quarterLabels)
quarterDict=dict(zip(quarterNames,range(len(quarterNames))))
y1=np.array([quarterDict[cl] for cl in quarterLabels])

#column department
departmentLabels=raw_data[:,2]
departmentNames=np.unique(departmentLabels)
departmentDict=dict(zip(departmentNames,range(len(departmentNames))))
y2=np.array([departmentDict[cl] for cl in departmentLabels])

#column department
