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
my_summary = pd.DataFrame(columns = ['type','station','train','valid','total'])

for worktype in types:
    for workstation in stations:
        print("Starting type and station: ", worktype, ' ', workstation)
        #estimation step
        y_train, X_train = process_data(pm_train, weather,worktype,workstation,temp_var,if_log)
        
        X_train_train = X_train.iloc[:int(np.round(len(X_train)*2/3)),:]
        y_train_train = y_train[:int(np.round(len(X_train)*2/3))]
        X_train_valid = X_train.iloc[int(np.round(len(X_train)*2/3)):,:]
        y_train_valid = y_train[int(np.round(len(X_train)*2/3)):]
            
        print("Estimation ------------------: ")
        try:        
            my_model = my_estimate(X_train_train,y_train_train)
            if if_log:
                y_hat= np.exp(my_model.predict(X_train_train))
            else:
                y_hat= my_model.predict(X_train_train)
        except:
            print('No data in train')
            continue
        
        #training set
        print("On training set:")
        on_train = 0
        if if_log:
            print(np.sqrt(mean_squared_error(np.exp(y_train_train),y_hat)))
            on_train = np.sqrt(mean_squared_error(np.exp(y_train_train),y_hat))
        else:
            print(np.sqrt(mean_squared_error(y_train_train,y_hat)))
            on_train = np.sqrt(mean_squared_error(y_train_train,y_hat))
            
        #validation set
        print("On validation set:")
        on_valid = 0
        if if_log:
            y_hat_valid= np.exp(my_model.predict(X_train_valid))
            print(np.sqrt(mean_squared_error(np.exp(y_train_valid),y_hat_valid)))
            on_valid = np.sqrt(mean_squared_error(np.exp(y_train_valid),y_hat_valid))
        else: 
            y_hat_valid= my_model.predict(X_train_valid)
            print(np.sqrt(mean_squared_error(y_train_valid,y_hat_valid)))       
            on_valid = np.sqrt(mean_squared_error(y_train_valid,y_hat_valid))
            
        print("On total set:")
        on_total = 0
        my_model = my_estimate(X_train,y_train)
        
        if if_log:
            y_hat_tot= np.exp(my_model.predict(X_train))
            print(np.sqrt(mean_squared_error(np.exp(y_train),y_hat_tot)))
            on_total = np.sqrt(mean_squared_error(np.exp(y_train),y_hat_tot))
        else:
            y_hat_tot= my_model.predict(X_train)
            print(np.sqrt(mean_squared_error(y_train,y_hat_tot)))
            on_total = np.sqrt(mean_squared_error(y_train,y_hat_tot))

        pm_train_c.loc[((pm_train_c['type']==worktype)&(pm_train_c['station']==workstation)),'y_test'] = y_hat_tot
        pm_train_c.loc[((pm_train_c['type']==worktype)&(pm_train_c['station']==workstation)),'error'] = y_hat_tot - pm_train_c.loc[((pm_train_c['type']==worktype)&(pm_train_c['station']==workstation)),'aqi']
            
        # prediction step
        y_test, X_test = process_data(pm_test, weather,worktype,workstation,temp_var,if_log)
        X_test         = test_add_prep(X_test,X_train)
        
        try:
            if if_log:
                y_test_hat = np.exp(my_model.predict(X_test))
            else:
                y_test_hat = my_model.predict(X_test)
        except:
            print('No data in test')
            continue
            
        pm_test.loc[((pm_test['type']==worktype)&(pm_test['station']==workstation)),'y_test'] = y_test_hat        

        item = [worktype,workstation,on_train,on_valid,on_total]
        my_summary.loc[len(my_summary)] = item


my_summary.to_csv('my_summary.csv',index=False)
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
submission.to_csv('submission_cat.csv',index=False)


print("from best submission:")
subB = pd.read_csv('sub6002.csv')
print(np.sqrt(mean_squared_error(subB['aqi'],pm_test['aqi'])))

print("from nan submission:")
sub_nan = pd.read_csv('sub_linear.csv')
print(np.sqrt(mean_squared_error(sub_nan['aqi'],pm_test['aqi'])))

