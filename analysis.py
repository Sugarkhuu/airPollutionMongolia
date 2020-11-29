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

s_dir = '/home/sugarkhuu/Documents/python/airPollutionMongolia'
#s_dir = "C:\\Users\\sugar\\Documents\\my\\py\\airPollutionMongolia"
os.chdir(s_dir)


from helper import *

pm_train = pd.read_csv('./ulaanbaatar-city-air-pollution-prediction/pm_train.csv')
pm_test = pd.read_csv('./ulaanbaatar-city-air-pollution-prediction/pm_test.csv')
weather = pd.read_csv('./ulaanbaatar-city-air-pollution-prediction/weather.csv')

stations = pm_train.station.unique()
types    = pm_train.type.unique()
worktype = types[0]
if_log = True

pm_train = initial_process(pm_train)
pm_test  = initial_process(pm_test)

temp_var = 'apparentTemperature'
weather  = process_weather(weather,temp_var)

#pm_train.loc[pm_train['aqi']>400,'aqi'] = 400
#pm_train = pm_train[pm_train['month'].isin(pm_test['month'].unique())]


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
    my_model = my_estimate(X_train_train,y_train_train)
    
    if if_log:
        y_hat= np.exp(my_model.predict(X_train_train))
    else:
        y_hat= my_model.predict(X_train_train)
    
    #training set
    print("On training set:")
    #print('Log-lin MSE On validation Data: {}'.format(np.sqrt(mean_squared_error(np.exp(y_train),y_hat))))
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
    my_model = my_estimate(X_train,y_train)
    
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
        y_test_hat = np.exp(my_model.predict(X_test))
    else:
        y_test_hat = my_model.predict(X_test)

    pm_test.loc[pm_test['type']==worktype,'y_test'] = y_test_hat        


print('Processing done.')

test_valid = pm_test.loc[~pm_test['aqi'].isnull(),'aqi'] 
test_hat   = pm_test.loc[~pm_test['aqi'].isnull(),'y_test'] 
print("From available:")
print(np.sqrt(mean_squared_error(test_valid,test_hat)))
plt.scatter(test_valid, test_hat)

# post-process
pm_test['aqi_back'] = pm_test['aqi']
pm_test['error']    = pm_test['y_test'] - pm_test['aqi']
pm_test.loc[pm_test['aqi'].isnull(),'aqi'] = pm_test.loc[pm_test['aqi'].isnull(),'y_test'] 

submission = pm_test[['ID','aqi']].copy()
assert submission['aqi'].isnull().sum() == 0
submission.to_csv('submission.csv',index=False)


print("from last submission:")
best = pd.read_csv('submission667.csv')
best = best.rename(columns={'aqi': "aqi_best"})
pm_test = pm_test.merge(best,on='ID',how = 'left')
print(np.sqrt(mean_squared_error(submission['aqi'],best['aqi_best'])))



largess

plt.scatter(old['aqi'],submission['aqi'])
plt.figure()
old['aqi'].hist(bins=100)
submission['aqi'].hist(bins=100)

a=pd.DataFrame()
a['old'] = old['aqi']
a['sub'] = submission['aqi']

a[['sub','old']].plot()

# IDEAS: try winter by sth else
# minimum of last day
# explore peaks and get dummy for them
# wind speed and difference in stations
# difference between temp and apparTemp

# should be careful about humidity as it has outliers in winter time
# temperature variables could be in logs
# a month can be divided into two parts: early, late

#https://towardsdatascience.com/recurrent-neural-networks-by-example-in-python-ffd204f99470
#https://towardsdatascience.com/my-secret-sauce-to-be-in-top-2-of-a-kaggle-competition-57cff0677d3c


test_avg = pm_test.iloc[65000:][pm_test['month'].isin([3,10,11])].groupby(['month','hour'])['aqi'].mean()
train_avg = pm_train[pm_train['month'].isin([3,10,11])].groupby(['month','hour'])['aqi'].mean()

test_avg.plot()
train_avg.plot()

weather['day'] = weather['date'].dt.strftime('%Y-%m-%d')
weather['year'] = pd.DatetimeIndex(weather['date']).year
weather['month'] = pd.DatetimeIndex(weather['date']).month
weather['hour'] = pd.DatetimeIndex(weather['date']).hour



pm_train = pm_train_c.copy()
weather = pd.read_csv('./ulaanbaatar-city-air-pollution-prediction/weather.csv')
weather['date'] = pd.to_datetime(weather['date'])
pm_train = pm_train.merge(weather,on='date',how='left')



id_station = 10
id_type    = 1
sdate   = '2016-10-01'
edate   = '2017-10-31'

short = pm_train[pm_train['station']==stations[id_station]]
short = short[short['type']==types[id_type]]
short = short[short['day']>=sdate]
short = short[short['day']<=edate]

short['temperature'] = short['temperature']*10
short['apparentTemperature'] = short['apparentTemperature']*10+500
short['windSpeed'] = short['windSpeed']*100+200
short.set_index('date')[['aqi','y_test']].plot()
short[['date','aqi','y_test','temperature','apparentTemperature','dewPoint','humidity']]
#short[['error','temperature']].plot()




studySet = pm_train[pm_train['error']<-300]

#studySet['windSpeed'].hist(bins=100) equal or below 2 (like usual)
pm_train['windSpeed'].hist(bins=100) 
#studySet['humidity'].hist(bins=100) 0.65-0.75 (like usual)
plt.scatter(studySet['windSpeed'],studySet['humidity'])
plt.scatter(pm_train['windSpeed'],pm_train['humidity'])

studySet['temperature'].hist(bins=100) # below -15 mostly
studySet['month'].hist(bins=100) # below -15 mostly
studySet['hour'].hist(bins=100) # below -15 mostly
plt.scatter(studySet['month'],studySet['hour'])


a = pm_test[~pm_test['aqi'].isnull()]

plt.figure()
sns.boxplot(y='aqi', x='month', 
                 data=a, 
                 palette="colorblind",
                 hue='station')



pm_train['temperature'].hist(bins=100)



wstart = 9
wend   = 5
pm_train['winter']=0
pm_train.loc[~((pm_train['month']>=wend+1)&(pm_train['month']<=wstart-1)),'winter'] = 1    
pm_train_winter = pm_train[pm_train['winter']==0]    


sns.boxplot(y='aqi', x='month', 
                 data=pm_train, 
                 palette="colorblind",
                 hue='station')


plt.scatter(pm_train['aqi'],pm_train['error'])
plt.hist(pm_train_winter['error'],bins=100)



pm_above300 = pm_train[pm_train['aqi']>350]


# above 300 happening less on weekends - especially Sunday
pm_above300['dayofweek'].hist(bins=24)

# 1,12 peak, then 2,11
pm_above300['month'].hist(bins=24)

# two peaked. In the morning it stays longer, between day and night goes down too much 
pm_above300['hour'].hist(bins=24)

# 2017 was better than 2016
pm_above300['year'].hist(bins=24)

# average error = -98.8
pm_above300['error'].mean()
pm_above300['error'].hist(bins=100)

# 2.5 is most
pm_above300['type'].hist()

# 
pm_above300[pm_above300['type']==types[0]]['summary'].hist()

#np.sqrt(98.8*15697/132000)
#np.sqrt((98.8**2)*15697/132000)
#np.sqrt((400**2-300**2)*15697/132500) = 91
#np.sqrt((300**2-200**2)*15697/132500) = 76.9




short = pm_train.groupby(['year','month'])['aqi'].mean().reset_index()

plt.scatter(short['month'],short['aqi'],c=short['year'])

plt.scatter(pm_train['month'],pm_train['aqi'],c=pm_train['year'])
plt.scatter(weather['month'],weather['temperature'],c=weather['year'])










sns.boxplot(y='aqi', x='month', 
                 data=pm_train, 
                 palette="colorblind",
                 hue='year')

plt.figure()
sns.boxplot(y='aqi', x='month', 
                 data=pm_above300, 
                 palette="colorblind",
                 hue='year')

plt.figure()
sns.boxplot(y='windBearing', x='month', 
                 data=weather, 
                 palette="colorblind",
                 hue='year')

g = sns.FacetGrid(weather, row="year", col="month",margin_titles=True)
g.map(sns.regplot, "hour", "temperature", color=".3", fit_reg=False, x_jitter=.1)

g = sns.FacetGrid(pm_train, row="year", col="month", hue="type",margin_titles=True)
g.map(sns.regplot, "hour", "aqi", color=".3", fit_reg=False, x_jitter=.1)

                
exit()

#enc = OneHotEncoder(sparse=False)
#X_transform = enc.fit_transform(X)

pm_train = pd.read_csv('./ulaanbaatar-city-air-pollution-prediction/pm_train.csv')
pm_train['date'] = pd.to_datetime(pm_train['date'])
pm_train['day'] = pm_train['date'].dt.strftime('%Y-%m-%d')
pm_train['year'] = pd.DatetimeIndex(pm_train['date']).year
pm_train['month'] = pd.DatetimeIndex(pm_train['date']).month
pm_train['hour'] = pd.DatetimeIndex(pm_train['date']).hour
pm_train['dayofweek'] = pd.DatetimeIndex(pm_train['date']).dayofweek
pm_train['dayofmonth'] = pd.DatetimeIndex(pm_train['date']).day

pm_train = pm_train[pm_train['type']==worktype]
pm_train['y_hat'] = np.exp(my_model.predict(X_train))

month = 7
day = 25;15;14
year = 2018
station = 1;9;8;11;8;1;6
ntype = 0

pm_train['month'] = pd.DatetimeIndex(pm_train['date']).month
pm_train = pm_train.merge(weather,on='date',how='left')
pm_train['error']=pm_train['aqi']-pm_train['y_hat']
sample = pm_train[(pm_train['month']==month)&(pm_train['type']==worktype)]
sample = sample[sample['year']==year]
simple_draw = sample.groupby(['dayofmonth','hour'])[['aqi','y_hat']].mean().unstack(['dayofmonth'])


simple_draw.loc[:,('aqi',day)].plot()
simple_draw.loc[:,('y_hat',day)].plot()

weather['day'] = weather['date'].dt.strftime('%Y-%m-%d')
weather['month'] = pd.DatetimeIndex(weather['date']).month
weather['hour'] = pd.DatetimeIndex(weather['date']).hour
weather['year'] = pd.DatetimeIndex(weather['date']).year
short_w = weather[(weather['day']>='2017-06-05')&(weather['day']<='2017-08-17')]
short_w[['date','uvIndex']].set_index('date').plot()

#pm_train.groupby(['year','month','hour'])[['aqi','y_hat','error']].mean().unstack(['month']).plot()
pm_train.groupby(['month','hour'])[['error']].mean().unstack(['month']).plot()
pm_train.groupby(['month','hour'])[['cloudCover']].mean().unstack(['month']).plot()
weather.groupby(['year','month','hour'])[['apparentTemperature']].mean().unstack(['year','month']).plot()

plt.scatter(pm_train['error'],pm_train['winter_h12'])

#location of larges
#pm_train.groupby(['year','month','hour'])[['error']].mean().unstack(['year','month']).idxmax(axis=1)
#(pm_train.groupby(['year','month','hour'])[['error']].mean().unstack(['year','month'])>50).sum()

Index(['date', 'apparentTemperature', 'humidity', 'windSpeed', 'windBearing',
       'cloudCover', 'uvIndex', 'visibility', 'apparentTemperature_1',
       'windBearing_1', 'humidity_1', 'apparentTemperature_2', 'windBearing_2',
       'humidity_2', 'apparentTemperature_3', 'windBearing_3', 'humidity_3',
       'apparentTemperature_4', 'windBearing_4', 'humidity_4', 'day'],
      dtype='object')

### prediction phase

pm_test['winter']=0
pm_test.loc[~((pm_test['month']>=wend+1)&(pm_test['month']<=wstart-1)),'winter'] = 1

pm_test['dayhours']=0
pm_test.loc[((pm_test['hour']>=dstart)&(pm_test['hour']<=dend)),'dayhours'] = 1

pm_test['nighthours']=0
pm_test.loc[~((pm_test['hour']>=nend+1)&(pm_test['hour']<=nstart-1)),'nighthours'] = 1

#winter dayhours
#winter nighthours

pm_test['winter_dayhours'] = pm_test['winter']*pm_test['dayhours']
pm_test['winter_nighthours'] = pm_test['winter']*pm_test['nighthours']

#weather = weather.drop(['day', 'year', 'month', 'hour', 'dayofweek', 'dayofmonth'],axis=1)

pm_test = pm_test.merge(weather,on='date',how='left')
#pm_test['winter_temp'] = pm_test['winter']*pm_test['apparentTemperature']

#var_list = ['month','station','type','hour','dayofweek',
#            'precipIntensity','precipProbability', 'temperature', 'apparentTemperature', 'dewPoint',
#            'humidity', 'windSpeed', 'windBearing', 'cloudCover', 'uvIndex',
#            'visibility']
#
#pm_back_X = pm_back[var_list]
#pm_test_X = pm_test[var_list]

pm_test_backup = pm_test.copy()
pm_test = encoding(pm_test)

pm_test = pm_test[pm_test['type']==worktype]
X_test = pm_test.drop(['ID','year','date','latitude','longitude','aqi','dayofmonth','day','station'],axis=1)
X_test = X_test.drop(['dayhours','nighthours','winter','type'],axis=1)


missing_cols = np.setdiff1d(X_train.columns, X_test.columns)
X_test = pd.concat([X_test,pd.DataFrame(columns=missing_cols)])
X_test[missing_cols] = 0
X_test = X_test[X_train.columns]


#X_test.isna().any()[lambda x: x]
X_test = X_test.interpolate()

y_test= model.predict(X_test)

#pm_test = pm_test_backup
pm_test['y_test'] = np.exp(y_test)
df_mean = pm_train.groupby(['month','dayofmonth','hour','station','type']).agg({'aqi':['median']})
df_mean.columns = ['aqi_median']
df_mean = df_mean.reset_index()

pm_test = pm_test.merge(df_mean,on=['month','hour','station','type','dayofmonth'],how='left')

# to check how off 
pm_test['y_test'].hist(bins=100)
df_check = pm_test[~pm_test['aqi'].isnull()][['date','aqi','y_test']]
df_check['year'] = pd.DatetimeIndex(df_check['date']).year
df_check['day'] = pd.DatetimeIndex(df_check['date']).day
plt.scatter(df_check[df_check['year']==2020]['aqi'],df_check[df_check['year']==2020]['y_test'])
weird = pm_test[pm_test['y_test']<0]
weird['y_test'].hist(bins=100)


#pm_test.loc[((pm_test['y_test']<0)&~(pm_test['aqi_median'].isnull())),'y_test'] = pm_test.loc[((pm_test['y_test']<0)&~(pm_test['aqi_median'].isnull())),'aqi_median'] 
#pm_test.loc[(pm_test['y_test']<0),'y_test'] = 20

# combine result
pm_test0 = pm_test_backup[pm_test_backup['type']==types[0]]
pm_test1 = pm_test_backup[pm_test_backup['type']==types[1]]
pm_test0['y_test'] = np.exp(y_test0)
pm_test1['y_test'] = np.exp(y_test1)
pm_res = pd.concat([pm_test0,pm_test1])

pm_res.loc[pm_res['aqi'].isnull(),'aqi'] = pm_res.loc[pm_res['aqi'].isnull(),'y_test'] 



#y_test1 = y_test
#y_test0=y_test1
#y_test1 = y_test


submission = pm_res[['ID','aqi']].copy()
#submission.loc[submission['aqi']<0,'aqi'] = 20.0
#submission.loc[submission['aqi']>100,'aqi'] = submission.loc[submission['aqi']>100,'aqi']-40
submission.to_csv('submission.csv',index=False)

old = pd.read_csv('submission843.csv')
plt.scatter(old['aqi'],submission['aqi'])
old['aqi'].hist(bins=100)
plt.figure()
submission['aqi'].hist(bins=100)


print(df)
nan_values = df. isna()
nan_columns = nan_values. any()
columns_with_nan = df. columns[nan_columns]. tolist()
print(columns_with_nan)




#plt.scatter(weather['temperature'],weather['dewPoint'],c=weather['month'])
#
#small = pm_back[~((pm_back['month']>=4)&(pm_back['month']<=9))]
#small = pm_back[((pm_back['month']>=4)&(pm_back['month']<=9))]
#plt.scatter(small['temperature'],small['aqi'],c=small['month'])
#plt.scatter(small['temperature'],small['windBearing'],c=small['month'])

#pm_back.groupby(['year','month','dayofmonth','hour'])['aqi'].mean().unstack(['year','month','dayofmonth']).loc[:,(slice(None),month,dayofmonth)].plot()
#pm_test1.groupby(['year','month','dayofmonth','hour'])['aqi'].mean().unstack(['year','month','dayofmonth']).loc[:,(slice(None),month,dayofmonth)].plot()
#
#pm_test1[['day','hour','station','aqi']][pm_test1['day']=='2019-01-05'].groupby(['station','hour'])['aqi'].mean().unstack('station').plot()














fig = plt.figure(figsize = (10, 7))
ax = plt.axes(projection ="3d")
#ax.scatter3D(draw['hour'], draw['aqi'], draw['temperature'], c= draw['dayofmonth'])
ax.scatter3D(draw['hour'], draw['aqi'], draw['temperature'])





pm_back  = pm_train.merge(weather, on='date',how='left')
small = pm_back[~((pm_back['month']>=4)&(pm_back['month']<=9))]


small_sorted = small.sort_values(['station','type','day','hour'])
small_sorted['aqi_1'] = small_sorted.groupby(['station','type'])['aqi'].diff(5)

plt.scatter(small_sorted['temp_1_2'],small_sorted['aqi_1'], c=small_sorted['hour'])


month = 1
sample = pm_back[pm_back['month']==month]

fig, ax = plt.subplots()
scatter = ax.scatter(sample['temperature'],sample[['aqi']],c=sample['hour'])
legend1 = ax.legend(*scatter.legend_elements(),
                    loc="lower left", title="Hours")
ax.add_artist(legend1)
plt.show()


month = 1
station = 1
ntype = 0
year = 2017



sample = pm_back[(pm_back['month']==month)&(pm_back['type']==types[ntype])]
sample = sample[sample['year']==year]
#sample.groupby(['year','dayofmonth','hour'])['aqi'].mean().unstack(['year','dayofmonth']).plot()
#sample.groupby(['year','dayofmonth','hour'])['temperature'].mean().unstack(['year','dayofmonth']).plot()
#sample.groupby(['year','dayofmonth','hour'])[['aqi','temperature']].mean().unstack(['year','dayofmonth']).plot()
simple_draw = sample.groupby(['year','dayofmonth','hour'])[['aqi']].mean().unstack(['dayofmonth'])
draw = sample.groupby(['year','dayofmonth','hour'])[['aqi','temperature']].mean().reset_index()

simple_draw.plot()


# Creating plot
ax.scatter3D(x, y, z, color = "green")
plt.title("simple 3D scatter plot")

plt.legend(sample['hour'].unique())
legend1 = ax.legend(*scatter.legend_elements(),
                    loc="lower left", title="Classes")



df_plot = pm_back[['month','hour','station','type','aqi']].groupby(['month','hour','station','type'])['aqi'].mean().unstack(['month','station','type'])
mean_by_month = df_plot.mean().reset_index().groupby('month')[0].mean()


main_mean = pm_back[['month','hour','station','type','dayofweek','aqi','temperature']].groupby(['month','hour','station','dayofweek','type']).agg({'aqi':['median'],'temperature':['median']})
main_mean.columns = ['aqi_mean','temperature_mean']
pm_back1 = pm_back.merge(main_mean,on=['month','hour','station','type','dayofweek'],how='left')
pm_back1['aqi_error'] = pm_back1['aqi'] - pm_back1['aqi_mean']
pm_back1['temperature_error'] = pm_back1['temperature'] - pm_back1['temperature_mean']


pm_test1 = pm_test.merge(main_mean,on=['month','hour','station','type','dayofweek'],how='left')

pm_test1['aqi_error'] = pm_test1['aqi'] - pm_test1['aqi_mean']
pm_test1['temp_error'] = pm_test1['temperature'] - pm_test1['temperature_mean']
df_aqi = pm_test1[~pm_test1['aqi_error'].isnull()]
df_aqi = df_aqi.set_index('date')

test_days = df_aqi['day'].unique()

df_test_day = df_aqi[df_aqi['day'] == test_days[2]]
df_test_day.aqi_error.mean()
df_test_day.aqi_error.hist()



riseIn201901 = pm_test1[(pm_test1['month']==1)&(~pm_test1['aqi_error'].isnull())][['station','aqi_error']].groupby('station')['aqi_error'].mean()
declineIn202010 = pm_test1[(pm_test1['month']==10)&(~pm_test1['aqi_error'].isnull())][['station','aqi_error']].groupby('station')['aqi_error'].mean()
declineIn202011 = pm_test1[(pm_test1['month']==11)&(~pm_test1['aqi_error'].isnull())][['station','aqi_error']].groupby('station')['aqi_error'].mean()

pm_test1[(pm_test1['month']==11)&(~pm_test1['aqi_error'].isnull())].groupby(['month','hour','station','type'])['aqi_error'].sum().unstack(['month','station','type']).plot()
pm_test1[(pm_test1['month']==10)&(~pm_test1['aqi_error'].isnull())].groupby(['month','hour','station','type'])['aqi_error'].median().unstack(['month','station','type']).plot()
pm_test1[(pm_test1['month']==1)&(~pm_test1['aqi_error'].isnull())].groupby(['month','hour','station','type'])['aqi_error'].median().unstack(['month','station','type']).plot()
pm_test1[(pm_test1['month']==11)&(~pm_test1['aqi_error'].isnull())&(pm_test1['hour']>20)].groupby(['day','hour','station','type'])['aqi_error'].median().unstack(['day','station','type']).plot()

n_station = 0
n_month   = 10
pm_test1[(pm_test1['station']==stations[n_station])&(pm_test1['month']==n_month)&(~pm_test1['aqi_error'].isnull())&(pm_test1['hour']>0)].groupby(['day','hour','station','type'])['aqi_error'].median().unstack(['day','station','type']).plot()
pm_test1[(pm_test1['station']==stations[n_station])&(pm_test1['month']==n_month)&(~pm_test1['aqi_error'].isnull())&(pm_test1['hour']>0)].groupby(['day','hour','station','type'])['temp_error'].median().unstack(['day','station','type']).plot()



### useful figure

df_plot = pm_back[['month','hour','station','type','aqi']].groupby(['month','hour','station','type'])['aqi'].mean().unstack(['month','station','type'])

figure, axes = plt.subplots(4, 3)

for i in range(4):
    for j in range(3):
        print(i, j)
        legend = None
        n_type = 0
        try:
            if i == 0 and j == 0:
                df_plot.loc[:,(slice(None),stations[i*3+j], types[n_type])].iloc[:,:].plot(ax=axes[i,j])
            else:
                df_plot.loc[:,(slice(None),stations[i*3+j], types[n_type])].iloc[:,:].plot(ax=axes[i,j],legend=legend)
            axes[i,j].set_title(stations[i*3+j])
        except:
            axes[i,j].set_title(stations[i*3+j])
            print('No data')

figure.show()






pm_back1 = pm_back.merge(main_mean,on=['month','hour','station','type','dayofweek'],how='left')


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

reg = LinearRegression().fit(pm_back1['temperature_error'].fillna(0).values.reshape(-1,1),pm_back1['aqi_error'].fillna(0).values.reshape(-1,1))
y_hat = reg.predict(pm_back1['temperature_error'].fillna(0).values.reshape(-1,1))

pm_back1['aqi_temp_err'] = y_hat 
pm_back1['aqi_error1'] = pm_back1['aqi_error'] - pm_back1['aqi_temp_err'] 

plt.scatter(pm_back1['temperature_error'],pm_back1['aqi_error1'])
plt.scatter(pm_back1['windSpeed'],pm_back1['aqi_error'])
plt.scatter(pm_back1[pm_back1.station == stations[6]]['windBearing'],pm_back1[pm_back1.station == stations[6]]['aqi_error'])




df_plot = pm_back1[['day','month','hour','station','type','aqi_error']].groupby(['day','month','hour','station','type'])['aqi_error'].mean().unstack(['day','month','station','type'])

station = 5
n_type = 0
n_month = 8;7;6;5

df_plot.loc[:,(slice(None),n_month,stations[station], types[n_type])].iloc[:,:].plot()


pm_back1[['windSpeed_meanadj']] = pm_back1[['windSpeed']] - pm_back1[['windSpeed']].mean()

np.sqrt(mean_squared_error(y_valid,y_hat))










df_plot = pm_back[['month','hour','station','type','error']].groupby(['month','hour','station','type'])['error'].mean().unstack(['month','station','type'])

figure, axes = plt.subplots(4, 3)

for i in range(4):
    for j in range(3):
        print(i, j)
        legend = None
        n_type = 0
        try:
            if i == 0 and j == 0:
                df_plot.loc[:,(slice(None),stations[i*3+j], types[n_type])].iloc[:,:].plot(ax=axes[i,j])
            else:
                df_plot.loc[:,(slice(None),stations[i*3+j], types[n_type])].iloc[:,:].plot(ax=axes[i,j],legend=legend)
            axes[i,j].set_title(stations[i*3+j])
        except:
            axes[i,j].set_title(stations[i*3+j])
            print('No data')

figure.show()


df_plot.loc[:,(slice(None),stations[i*4+j], types[0])].iloc[:,:]


df_plot.loc[('2015-09-01'),(slice(None), 'PM10')].iloc[-100:,:].plot()

max_10 = df_plot.loc[:,(slice(None), 'PM10')].max(axis=1)
min_10 = df_plot.loc[:,(slice(None), 'PM10')].min(axis=1)

(max_10-min_10)[(max_10-min_10)==748.0].index


df_plot.loc[(max_10-min_10)[(max_10-min_10)>300].index,(slice(None), 'PM10')].iloc[-100:,:].plot()

#from sklearn.ensemble import AdaBoostClassifier
#from sklearn.tree import DecisionTreeClassifier
#tree = AdaBoostClassifier(DecisionTreeClassifier(), n_estimators=30, learning_rate=.1)


pm_back["type"] = pm_back["type"].astype('category')
pm_back["type"] =  pm_back["type"].cat.codes

pm_test["type"] = pm_test["type"].astype('category')
pm_test["type"] = pm_test["type"].cat.codes


def encoding(df):
    df = pd.get_dummies(df, columns=['month'], prefix='month')
    df = pd.get_dummies(df, columns=['hour'], prefix='hour')
    df = pd.get_dummies(df, columns=['dayofweek'], prefix='dayofweek')
    df = pd.get_dummies(df, columns=['station'], prefix='stat')
    df = pd.get_dummies(df, columns=['type'], prefix='type')
    return df



var_list = ['month','station','type','hour','dayofweek','temperature', 'dewPoint',
       'windSpeed', 'windBearing','visibility']

pm_back_X = pm_back[var_list]
pm_test_X = pm_test[var_list]

pm_back_X = encoding(pm_back_X)
pm_test_X = encoding(pm_test_X)



from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

n_train = int(np.round(len(pm_back_X)*5/10))


X_train = pm_back_X.loc[:n_train,:].fillna(0).values
y_train = pm_back.loc[:n_train,:]['aqi'].fillna(50)

X_valid = pm_back_X.loc[n_train:,:].fillna(0).values
y_valid = pm_back.loc[n_train:,:]['aqi'].fillna(50)



reg = LinearRegression().fit(X_train,y_train)
y_hat = reg.predict(X_valid)

np.sqrt(mean_squared_error(y_valid,y_hat))





missing_cols = np.setdiff1d(pm_back_X.columns, pm_test_X.columns)
pm_test_X = pd.concat([pm_test_X,pd.DataFrame(columns=missing_cols)])
pm_test_X[missing_cols] = 0
pm_test_X = pm_test_X[pm_back_X.columns]
X_test = pm_test_X.fillna(0).values

#X_train["station"] = X_train["station"].astype('category')
#X_train["station"] =  X_train["station"].cat.codes

#from sklearn.linear_model import LogisticRegression
## Create logistic regression object
#log_regr = LogisticRegression()
#log_regr.fit(X_train.values, y_train)



y_test = reg.predict(X_test)
pm_test['aqi'] = y_test -30


submission = pm_test[['ID','aqi']].copy()
submission.loc[submission['aqi']<0,'aqi'] = 20.0
submission.to_csv('submission.csv',index=False)


weather['date'] = pd.to_datetime(weather['date'])

pm_train['date'] = pd.to_datetime(pm_train['date'])
pm_back  = pm_train.merge(weather, on='date',how='left')
pm_back['day'] = pm_back['date'].dt.strftime('%Y-%m-%d')
pm_back['month'] = pd.DatetimeIndex(pm_back['date']).month
pm_back['hour'] = pd.DatetimeIndex(pm_back['date']).hour



from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
tree = AdaBoostClassifier(DecisionTreeClassifier(), n_estimators=300, learning_rate=.1)

X_train = pm_back[['month','hour','station','type','summary', 'icon', 'precipIntensity',
       'precipProbability', 'temperature', 'apparentTemperature', 'dewPoint',
       'humidity', 'windSpeed', 'windBearing', 'cloudCover', 'uvIndex',
       'visibility','temp_dev']]
y_train = pm_back['aqi']

X_test = pm_united[['month','hour','station','type','summary', 'icon', 'precipIntensity',
       'precipProbability', 'temperature', 'apparentTemperature', 'dewPoint',
       'humidity', 'windSpeed', 'windBearing', 'cloudCover', 'uvIndex',
       'visibility','temp_dev']]

X_train = X_train.fillna(0)
y_train = y_train.fillna(100)

X_train["station"] = X_train["station"].astype('category')
X_train["station"] =  X_train["station"].cat.codes

X_train["summary"] = X_train["summary"].astype('category')
X_train["summary"] =  X_train["summary"].cat.codes

X_train["icon"] = X_train["icon"].astype('category')
X_train["icon"] =  X_train["icon"].cat.codes

X_train["type"] = X_train["type"].astype('category')
X_train["type"] =  X_train["type"].cat.codes

tree.fit(X_train, y_train)

tree_pred = tree.predict(X_test)





a = pm_back.groupby(['month','hour','station','type'])['aqi_dev','temp_dev'].mean()
a.columns = ['aqi_dev','temp_dev']
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

