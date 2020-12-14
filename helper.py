import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from catboost import CatBoostRegressor
from xgboost import XGBRegressor

def process_data(df, df_weather,worktype,temp_var,if_log=True):
    winterStart = 11;11
    winterEnd   = 2;2

    summerStart = 5;6
    summerEnd   = 9
    
    deepWinterStart = 12
    deepWinterEnd   = 1
    
    nightHStart = 20;21;20
    nightHEnd   = 5;3;4

    dayHStart = 10;10;9
    dayHEnd   = 14;15;16

    df['winter']=0
    df.loc[~((df['month']>=winterEnd+1)&(df['month']<=winterStart-1)),'winter'] = 1    
    

    df['deepWinter']=0
    df.loc[~((df['month']>=deepWinterEnd+1)&(df['month']<=deepWinterStart-1)),'deepWinter'] = 1    
    

    df['summer']=0
    df.loc[((df['month']>=summerStart)&(df['month']<=summerEnd)),'summer'] = 1
    
   
    df = encoding(df)
    df = df.merge(df_weather,on='date',how='left')   
        
    for hour in np.linspace(dayHStart,dayHEnd,dayHEnd-dayHStart+1):
        df['winter_h' + str(int(hour))]  = df['winter']*df['hour_' + str(int(hour))]
        df['deepWinter_h' + str(int(hour))]  = df['deepWinter']*df['hour_' + str(int(hour))]        
        for i in range(1):
            if i != 0:
                df['winter_hh_temp' + '_' + str(i) + '_' + str(hour)] = df['winter']*df['hour_' + str(int(hour))]*df[temp_var + '_' + str(i)]
            else:
                df['winter_hh_temp' + '_' + str(hour)] = df['winter']*df['hour_' + str(int(hour))]*df[temp_var]


    for hour in np.linspace(nightHStart,24,24-nightHStart+1):
        df['winter_h' + str(int(hour))]  = df['winter']*df['hour_' + str(int(hour))]
        df['deepWinter_h' + str(int(hour))]  = df['deepWinter']*df['hour_' + str(int(hour))]       
        for i in range(1):
            if i != 0:
                df['winter_hh_temp' + '_' + str(i) + '_' + str(hour)] = df['winter']*df['hour_' + str(int(hour))]*df[temp_var + '_' + str(i)]
            else:
                df['winter_hh_temp' + '_' + str(hour)] = df['winter']*df['hour_' + str(int(hour))]*df[temp_var]        
    for hour in np.linspace(1,nightHEnd,nightHEnd-1+1):
        df['winter_h' + str(int(hour))]  = df['winter']*df['hour_' + str(int(hour))]
        df['deepWinter_h' + str(int(hour))]  = df['deepWinter']*df['hour_' + str(int(hour))]    
        for i in range(1):
            if i != 0:
                df['winter_hh_temp' + '_' + str(i) + '_' + str(hour)] = df['winter']*df['hour_' + str(int(hour))]*df[temp_var + '_' + str(i)]
            else:
                df['winter_hh_temp' + '_' + str(hour)] = df['winter']*df['hour_' + str(int(hour))]*df[temp_var]
           
   
    lagvars = [temp_var,'windSpeed','humidity'] 
    
    for var in lagvars:
        
        for i in range(5):
            if i !=0:
                df['winter' + var + '_' + str(i)] = df['winter']*df[var + '_' + str(i)]
                if var == 'humidity':   
                    df['summer' + var + '_' + str(i)] = df['summer']*df[var + '_' + str(i)]
            else:
                df['winter' + var] = df['winter']*df[var]
                if var == 'humidity':   
                    df['summer' + var] = df['summer']*df[var]
            
    df=df.interpolate()
    df=df.interpolate(method='backfill')
    
    df.loc[df['aqi']<1,'aqi'] = 1 
    df['l_aqi'] = np.log(df['aqi'])
    
    df = df[df['type']==worktype]
    
    if if_log:
        Y = df.loc[:,'l_aqi']
    else:
        Y = df.loc[:,'aqi']
    
    X = df.drop(['ID','year','date','latitude','longitude','aqi','l_aqi','dayofmonth','day','source'],axis=1)
    X = X.drop(['winter','summer','type','month'],axis=1)
    
    
    return Y, X

def encoding(df):
    df = pd.get_dummies(df, columns=['hour'], prefix='hour')
    df = df.rename(columns={'hour_0': "hour_24"})
    df = df.drop('hour_6',axis=1)
    df = pd.get_dummies(df, columns=['station'], prefix='station')
    return df


def initial_process(df):
    df['date'] = pd.to_datetime(df['date'])
    df['day'] = df['date'].dt.strftime('%Y-%m-%d')
    df['year'] = pd.DatetimeIndex(df['date']).year
    df['month'] = pd.DatetimeIndex(df['date']).month
    df['hour'] = pd.DatetimeIndex(df['date']).hour
    df['dayofmonth'] = pd.DatetimeIndex(df['date']).day
    return df

def process_weather(weather,temp_var):
    weather['date'] = pd.to_datetime(weather['date'])
    weather['day'] = weather['date'].dt.strftime('%Y-%m-%d')
    weather['year'] = pd.DatetimeIndex(weather['date']).year
    weather['month'] = pd.DatetimeIndex(weather['date']).month
    wavg = weather.groupby(['year'])['windSpeed'].mean().reset_index()
    wstd = weather.groupby(['year'])['windSpeed'].std().reset_index()
    wavg = wavg.rename(columns={'windSpeed': "windSpeed_year_avg"})
    wstd = wstd.rename(columns={'windSpeed': "windSpeed_year_std"})
    weather = weather.merge(wavg,on=['year'],how='left')
    weather = weather.merge(wstd,on=['year'],how='left')
    weather['wshl'] = 0
    weather.loc[((weather['windSpeed']<1)),'wshl'] = 1
    weather['windSpeed'] = (weather['windSpeed'] - weather["windSpeed_year_avg"])/weather["windSpeed_year_std"]
    
    weather = pd.get_dummies(weather, columns=['icon'], prefix='icon', drop_first=True)
    weather = weather.drop(['Unnamed: 0', 'summary','year','month','windSpeed_year_avg','windSpeed_year_std'],axis=1)
    weather = weather.interpolate()
    weather = weather.sort_values(by='date')
    
    whelp =pd.DataFrame()
    whelp['temp'] = weather.groupby(weather['day'])[temp_var].mean()
    whelp['temp_1'] = weather.groupby(weather['day'])[temp_var].mean().shift()
    whelp['temp_diff_1'] = weather.groupby(weather['day'])[temp_var].mean().diff()
    whelp['temp_2'] = weather.groupby(weather['day'])[temp_var].mean().shift(2)
    whelp['temp_diff_2'] = weather.groupby(weather['day'])[temp_var].mean().diff(2)
    whelp['temp_min'] = weather.groupby('day')['temperature'].min()
    whelp['temp_min_1'] = weather.groupby('day')['temperature'].min().shift()
    whelp['dayHumidity'] = weather.groupby(weather['day'])['humidity'].mean()
    whelp['dayuvIndex'] = weather.groupby(weather['day'])['uvIndex'].mean()
    whelp['daywindSpeed'] = weather.groupby(weather['day'])['windSpeed'].mean()
            
    weather = weather.merge(whelp,on='day',how='left')
    
    weather = weather.interpolate(method='backfill')
    weather = weather.drop(['day','dewPoint','precipProbability','precipIntensity'],axis=1)    
    if temp_var == 'apparentTemperature':
        weather = weather.drop(['temperature'],axis=1) 
    else:
        weather = weather.drop(['apparentTemperature'],axis=1) 
    
    lagvars = [temp_var,'windSpeed','humidity']
    for i in range(1,5):
        for var in lagvars:
            weather[var+'_'+str(i)] = weather[var].diff(i)

    return weather


def test_add_prep(df_test,df_train):
    missing_cols = np.setdiff1d(df_train.columns, df_test.columns)
    df_test = pd.concat([df_test,pd.DataFrame(columns=missing_cols)])
    df_test[missing_cols] = 0
    df_test = df_test[df_train.columns]
    df_test = df_test.interpolate()
    return df_test

def my_estimate(X,Y):
    
    run_model = 'lin';'xg';'cat';
    
    if run_model == 'lin':
        model = LinearRegression()
    elif run_model == 'cat':        
        model = CatBoostRegressor(iterations=1000,
                                 learning_rate=0.01,
                                 depth=7,
                                 eval_metric='RMSE',
                                 random_seed = 23,
                                 bagging_temperature = 0.1,
                                 od_type='Iter',
                                 metric_period = 75,
                                 od_wait=100)
    elif run_model == 'xg':        
        model = XGBRegressor(max_depth=10,learning_rate=0.07,n_estimators=500
                         ,sub_sample=0.6,gamma=1,colsample_bytree=0.5)
        
    print('Running model: ', run_model)
    
    model.fit(X, Y)
#    for i in range(len(X.columns)):
#        print(X.columns[i],model.coef_[i])
    return model
    
