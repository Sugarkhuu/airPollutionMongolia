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
    my_model = my_estimate(X_train_train,y_train_train)
    
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


print("from best submission:")
subB = pd.read_csv('sub606.csv')
print(np.sqrt(mean_squared_error(subB['aqi'],pm_test['aqi'])))


plt.scatter(subB['aqi'],pm_test['aqi'])
subB['aqi'].plot()
pm_test['aqi'].plot()

subB['aqi'].hist(bins=100)
pm_test['aqi'].hist(bins=100)



cat_depth7 = pd.read_csv('submission_catdepth7.csv')
lin_6317 = pd.read_csv('sub6317.csv')

plt.scatter(cat_depth7['aqi'],lin_6317['aqi'])

lin_6317['aqi'].plot()
cat_depth7['aqi'].plot()

avg = lin_6317.copy()
avg['aqi'] = (lin_6317['aqi'] + cat_depth7['aqi'])/2
avg.loc[30000:,'aqi'] = cat_depth7.loc[30000:,'aqi']


plt.scatter(lin_6317['aqi'],avg['aqi'])

lin_6317['aqi'].plot()
submission['aqi'].plot()
plt.title('<30000 is cat50:lin50, >30000 lin50')
