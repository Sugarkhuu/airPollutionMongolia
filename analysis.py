import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os


from sklearn.linear_model import LinearRegression


sugar_dir = '/home/sugarkhuu/Documents/python/airPollutionMongolia'
os.chdir(sugar_dir)


pm_train = pd.read_csv('./ulaanbaatar-city-air-pollution-prediction/pm_train.csv')
pm_test = pd.read_csv('./ulaanbaatar-city-air-pollution-prediction/pm_test.csv')
weather = pd.read_csv('./ulaanbaatar-city-air-pollution-prediction/weather.csv')

weather['date'] = pd.to_datetime(weather['date'])

pm_train['date'] = pd.to_datetime(pm_train['date'])
pm_back  = pm_train.merge(weather, on='date',how='left')
pm_back['day'] = pm_back['date'].dt.strftime('%Y-%m-%d')
pm_back['month'] = pd.DatetimeIndex(pm_back['date']).month
pm_back['hour'] = pd.DatetimeIndex(pm_back['date']).hour


a = pm_back.groupby(['month','hour','station','type'])['aqi','temperature'].mean()
a.columns = ['aqi_m','temp_m']
a = a.reset_index()
pm_back = pm_back.merge(a,on=['month','hour','station','type'],how='left')

pm_back['aqi_dev'] = pm_back['aqi'] - pm_back['aqi_m']
pm_back['temp_dev'] = pm_back['temperature'] - pm_back['temp_m']


#pm_back.plot(kind='scatter',x='temp_dev',y='aqi_dev')

model = LinearRegression()
model.fit(pm_back['temp_dev'].fillna(0).values.reshape(-1,1), pm_back['aqi_dev'].fillna(0).values)

my_coef = model.coef_[0]

pm_test['date'] = pd.to_datetime(pm_test['date'])
pm_test['day'] = pm_test['date'].dt.strftime('%Y-%m-%d')
pm_test['month'] = pd.DatetimeIndex(pm_test['date']).month
pm_test['hour'] = pd.DatetimeIndex(pm_test['date']).hour


pm_united = pm_test.merge(a,on=['month','hour','station','type'],how='left')
pm_united = pm_united.merge(weather,on='date',how='left')

pm_united['temp_dev'] = pm_united['temperature'] - pm_united['temp_m']
pm_united['aqi'] = pm_united['aqi_m'] + my_coef*pm_united['temp_dev'].fillna(0)

pm_united[pm_united['aqi'].isnull()]


#def get_result(df):
#    if df['aqi_x'] > 0:
#        a = df['aqi_x']
#    elif df['aqi_y'] > 0:
#        a = df['aqi_y']
#    else:
#        a=0
#    return a

#pm_united['aqi'] = pm_united.apply(get_result, axis=1)


submission = pm_united[['ID','aqi']]
submission.to_csv('submission.csv',index=False)

