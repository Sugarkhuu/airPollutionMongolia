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
pm_back['dayofweek'] = pd.DatetimeIndex(pm_back['date']).dayofweek

pm_test['date'] = pd.to_datetime(pm_test['date'])
pm_test = pm_test.merge(weather,on='date',how='left')
pm_test['day'] = pm_test['date'].dt.strftime('%Y-%m-%d')
pm_test['month'] = pd.DatetimeIndex(pm_test['date']).month
pm_test['hour'] = pd.DatetimeIndex(pm_test['date']).hour
pm_test['dayofweek'] = pd.DatetimeIndex(pm_test['date']).dayofweek


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

