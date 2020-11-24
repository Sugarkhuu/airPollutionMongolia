import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os


sugar_dir = '/home/sugarkhuu/Documents/python/airPollutionMongolia'
os.chdir(sugar_dir)


pm_train = pd.read_csv('./ulaanbaatar-city-air-pollution-prediction/pm_train.csv')
pm_test = pd.read_csv('./ulaanbaatar-city-air-pollution-prediction/pm_test.csv')
weather = pd.read_csv('./ulaanbaatar-city-air-pollution-prediction/weather.csv')

# meine Analyse
station_train = pm_train.station.unique()
station_test  = pm_test.station.unique()

stationsNotInTest = list(set(station_train) - set(station_test))

cols_train = pm_train.columns
cols_test  = pm_test.columns

colsNotInTest = list(set(cols_train) - set(cols_test))

dfs = [pm_train, pm_test, weather]

for df in dfs:
    df['date'] = pd.to_datetime(df['date'])
    df['year_month'] = df['date'].dt.strftime('%Y-%m')
    
    
train_count = pm_train.groupby(['year_month']).size().reset_index(name='counts')
test_count = pm_test.groupby(['year_month']).size().reset_index(name='counts')
weather_count = weather.groupby(['year_month']).size().reset_index(name='counts')

train_count.set_index('year_month',inplace=True)
test_count.set_index('year_month',inplace=True)
weather_count.set_index('year_month',inplace=True)

def time_series_graph(df, key='counts'):
    fig, ax = plt.subplots(figsize=(15, 6))
    
    # Add x-axis and y-axis
    ax.bar(df.index.values,
           df[key])

    ax.set(xlabel="Date",
           ylabel=key)

    plt.xticks(rotation=90)
    plt.show()
    
def get_norm(df, variable):
    var_mean = df[[variable]].mean().values
    var_std  = df[[variable]].std().values
    df[variable + '_norm'] = (df[variable]-var_mean)/var_std
    return df

pm_train = get_norm(pm_train, 'aqi')
weather  = get_norm(weather, 'temperature')
temperature = weather[['date','temperature_norm']]
pm_train = pm_train.merge(temperature, on='date',how='left')
pm_test  = pm_test.merge(temperature, on='date',how='left')

aqi_mean = pm_train['aqi'].mean()
aqi_std  = pm_train['aqi'].std()



stations = pm_train.station.unique().tolist()
types    = pm_train.type.unique().tolist()


from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=2)
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()

# PM10
Y =  pm_train[['aqi_norm']][pm_train['type'] == types[0]]
X =  pm_train[['temperature_norm']][pm_train['type'] == types[0]]
X = X.fillna(0)
Y = Y.fillna(0)
X_poly = poly_reg.fit_transform(X)
pol_reg = LinearRegression()
pol_reg.fit(X_poly, Y)

# PM2.5
Y_ =  pm_train[['aqi_norm']][pm_train['type'] == types[1]]
X_ =  pm_train[['temperature_norm']][pm_train['type'] == types[1]]
X_ = X_.fillna(0)
Y_ = Y_.fillna(0)
X_poly_ = poly_reg.fit_transform(X_)
pol_reg_ = LinearRegression()
pol_reg_.fit(X_poly_, Y_)


#predict
X_10 = pm_test[['temperature_norm']][pm_train['type'] == types[0]]
X_025 = pm_test[['temperature_norm']][pm_train['type'] == types[1]]
X_10 = X_10.fillna(0)
X_025 = X_025.fillna(0)

Y_10 = pol_reg.predict(poly_reg.fit_transform(X_10))
Y_025 = pol_reg_.predict(poly_reg.fit_transform(X_025))

Y_10_n  = aqi_mean + aqi_std*Y_10
Y_025_n = aqi_mean + aqi_std*Y_025

import matplotlib.pyplot as plt
plt.plot(Y_10_n)
plt.plot(Y_025_n)

insert_10 = X_10.copy()
insert_025 = X_025.copy()
insert_10['Y'] = Y_10_n
insert_025['Y'] = Y_025_n 

insert_10 = insert_10.reset_index()
insert_10['aqi'] = insert_10['Y']
insert_025 = insert_025.reset_index()
insert_025['aqi'] = insert_025['Y']
pm_test = pm_test.reset_index()


pm_test = pm_test.merge(insert_10,on='index',how='left')
pm_test = pm_test.merge(insert_025,on='index',how='left')
pm_test['aqi_proj'] = pm_test['aqi_y'].fillna(0) + pm_test['aqi'].fillna(0)

def fill_aqi(df):
   if df['aqi_x'] == np.nan:
       df['aqi_x'] = df['aqi_proj']
   else:
       df['aqi_x'] = df['aqi_x']
   return df

pm_test = pm_test.apply(fill_aqi,axis=1)

submission = pm_test[['ID','aqi_proj']]
submission.columns = ['ID','aqi']
submission.to_csv('submission.csv',index=False)
