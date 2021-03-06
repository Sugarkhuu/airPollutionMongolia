#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 19:32:55 2020

@author: sugarkhuu
"""

# Ensemble of Multiple Regression and CatBoost
# Air pollution prediction Mongolia

s_dir = '/home/sugarkhuu/Documents/python/airPollutionMongolia'
#s_dir = "C:\\Users\\sugar\\Documents\\my\\py\\airPollutionMongolia"


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import os

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

from mpl_toolkits import mplot3d
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


os.chdir(s_dir)

from helper import *

pm_train = pd.read_csv('./ulaanbaatar-city-air-pollution-prediction/pm_train.csv')
pm_test = pd.read_csv('./ulaanbaatar-city-air-pollution-prediction/pm_test.csv')
weather = pd.read_csv('./ulaanbaatar-city-air-pollution-prediction/weather.csv')

stations = pm_train.station.unique()
types    = pm_train.type.unique()
worktype = types[0]
if_log = True
run_model = 'lin';'cat';'xg'

pm_train = initial_process(pm_train)
pm_test  = initial_process(pm_test)

# replace NA in pm_train.aqi by station, month, hour average
tmp = pm_train.groupby(['station','month','hour'])['aqi'].mean()
tmp=tmp.reset_index()
tmp = tmp.rename(columns={'aqi': "aqi_mean"})
pm_train = pm_train.merge(tmp,on=['station','month','hour'],how='left')
pm_train.loc[pm_train['aqi']==0,'aqi'] = pm_train.loc[pm_train['aqi']==0,'aqi_mean']
del pm_train['aqi_mean']

temp_var = 'apparentTemperature' # 64.1 in apparent was 64.3 in normal
weather  = process_weather(weather,temp_var)
pm_train = pm_train[pm_train['month'].isin(pm_test['month'].unique())]

pm_train_c = pm_train.copy()

for worktype in types:
    print("Starting type: ", worktype)
    #estimation step
    y_train, X_train = process_data(pm_train, weather,worktype,temp_var,if_log)
    
    X_train_train = X_train.iloc[:int(np.round(len(X_train)*2/3)),:]
    y_train_train = y_train[:int(np.round(len(X_train)*2/3))]
    X_train_valid = X_train.iloc[int(np.round(len(X_train)*2/3)):,:]
    y_train_valid = y_train[int(np.round(len(X_train)*2/3)):]
        
    print("Estimation ------------------: ")
    my_model = my_estimate(run_model,X_train_train,y_train_train)
    
    if if_log:
        y_hat= np.exp(my_model.predict(X_train_train))
    else:
        y_hat= my_model.predict(X_train_train)
    
    #training set
    print("On training set:")
    if if_log:
        print(np.sqrt(mean_squared_error(np.exp(y_train_train),y_hat)))
    else:
        print(np.sqrt(mean_squared_error(y_train_train,y_hat)))
        
    #validation set
    print("On validation set:")
    if if_log:
        y_hat_valid= np.exp(my_model.predict(X_train_valid))
        print(np.sqrt(mean_squared_error(np.exp(y_train_valid),y_hat_valid)))
    else: 
        y_hat_valid= my_model.predict(X_train_valid)
        print(np.sqrt(mean_squared_error(y_train_valid,y_hat_valid)))        
        
    print("On total set:")
    my_model     = my_estimate(run_model,X_train,y_train)
    my_model_cat = my_estimate('cat',X_train,y_train) # hot fix for last
    
    if if_log:
        y_hat_tot= np.exp(my_model.predict(X_train))
        print(np.sqrt(mean_squared_error(np.exp(y_train),y_hat_tot)))
    else:
        y_hat_tot= my_model.predict(X_train)
        print(np.sqrt(mean_squared_error(y_train,y_hat_tot)))

    pm_train_c.loc[pm_train_c['type']==worktype,'y_test'] = y_hat_tot
    pm_train_c.loc[pm_train_c['type']==worktype,'error'] = y_hat_tot - pm_train_c.loc[pm_train_c['type']==worktype,'aqi']
        
    # prediction step
    y_test, X_test = process_data(pm_test, weather,worktype,temp_var,if_log)
    X_test         = test_add_prep(X_test,X_train)
    
    if if_log:
        y_test_hat     = np.exp(my_model.predict(X_test))
        y_test_hat_cat = np.exp(my_model_cat.predict(X_test)) # hot fix for last

        
    else:
        y_test_hat = my_model.predict(X_test)

    pm_test.loc[pm_test['type']==worktype,'y_test'] = y_test_hat        
    pm_test.loc[pm_test['type']==worktype,'y_test_cat'] = y_test_hat_cat        


print('Processing done.')


# Now let's mix the models
lin_weight = 0.50
cat_weight = 0.50

linear = pm_test[['ID','y_test']]
cat    = pm_test[['ID','y_test_cat']]
linear.columns = ['ID','aqi']
cat.columns = ['ID','aqi']

my_sub = linear.copy()
my_sub['aqi'] = lin_weight*linear['aqi'] + cat_weight*cat['aqi']


###  A bit manual adjustments. 2019-2020 should present a structural break
tmp = pm_train.groupby(['station','month','hour'])['aqi'].mean()
tmp=tmp.reset_index()
tmp = tmp.rename(columns={'aqi': "aqi_mean"})
#del pm_test['aqi_mean']
pm_test = pm_test.merge(tmp,on=['station','month','hour'],how='left')

check = my_sub.copy()
check['aqi_mean_diff'] = my_sub['aqi'] - pm_test['aqi_mean']
check['aqi_mean_diff'].hist(bins=100)

corr_factor = 0.5;.75
corr_thres  =  25;50;25
#my_sub.loc[check['aqi_mean_diff']<-100,'aqi'] = my_sub.loc[check['aqi_mean_diff']<-100,'aqi'] + 15
my_sub.loc[check['aqi_mean_diff']>corr_thres,'aqi'] = my_sub.loc[check['aqi_mean_diff']>corr_thres,'aqi'] - corr_factor*check.loc[check['aqi_mean_diff']>corr_thres,'aqi_mean_diff']

# readjust for those already given
my_sub.loc[~pm_test['aqi'].isnull(),'aqi'] = pm_test.loc[~pm_test['aqi'].isnull(),'aqi']
###


#submission file
submission = my_sub[['ID','aqi']].copy()
assert submission['aqi'].isnull().sum() == 0
submission.to_csv('submission.csv',index=False)

#Check
print("from 60.02 submission, should be zero:")
subB = pd.read_csv('sub6002.csv')
print(np.sqrt(mean_squared_error(submission['aqi'],sub6002['aqi'])))