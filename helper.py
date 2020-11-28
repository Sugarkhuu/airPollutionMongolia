import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

def process_data(df, df_weather,worktype,temp_var,if_log=True):
    winterStart = 10
    winterEnd   = 3

    summerStart = 6
    summerEnd   = 9
    
    deepWinterStart = 12
    deepWinterEnd   = 1
    
    nightHStart = 20;21;20
    nightHEnd   = 4;3;4

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

        

    for hour in np.linspace(nightHStart,23,23-nightHStart+1):
        df['winter_h' + str(int(hour))]  = df['winter']*df['hour_' + str(int(hour))]
        df['deepWinter_h' + str(int(hour))]  = df['deepWinter']*df['hour_' + str(int(hour))]       
        df['winter_h' + str(int(hour)) + 'windSpeed']  = df['winter']*df['hour_' + str(int(hour))]*df['windSpeed']
        df['deepWinter_h' + str(int(hour)) + 'windSpeed']  = df['deepWinter']*df['hour_' + str(int(hour))]*df['windSpeed']
    for hour in np.linspace(1,nightHEnd,nightHEnd-1+1):
        df['winter_h' + str(int(hour))]  = df['winter']*df['hour_' + str(int(hour))]
        df['deepWinter_h' + str(int(hour))]  = df['deepWinter']*df['hour_' + str(int(hour))]    
        df['winter_h' + str(int(hour)) + 'windSpeed']  = df['winter']*df['hour_' + str(int(hour))]*df['windSpeed']
        df['deepWinter_h' + str(int(hour)) + 'windSpeed']  = df['deepWinter']*df['hour_' + str(int(hour))]*df['windSpeed']
            
   
    lagvars = [temp_var,'windSpeed','windBearing','humidity','uvIndex','visibility']
    
    for var in lagvars:
        
        for i in range(25):
            if i !=0:
                df['winter' + var + '_' + str(i)] = df['winter']*df[var + '_' + str(i)]
                df['deepWinter' + var + '_' + str(i)] = df['deepWinter']*df[var + '_' + str(i)]
                if var == 'humidity':   
                    df['summer' + var + '_' + str(i)] = df['summer']*df[var + '_' + str(i)]
            else:
                df['winter' + var] = df['winter']*df[var]
                df['deepWinter' + var] = df['deepWinter']*df[var]
                if var == 'humidity':   
                    df['summer' + var] = df['summer']*df[var]
            
            
    df['winter_Sunday'] = df['winter']*df['dayofweek_1']
    df['winter_Saturday'] = df['winter']*df['dayofweek_6']  
    df['deepWinter_Sunday'] = df['deepWinter']*df['dayofweek_1']
    df['deepWinter_Saturday'] = df['deepWinter']*df['dayofweek_6']  

    df['winter_dayHumidity'] = df['winter']*df['dayHumidity']
    df['winter_dayuvIndex'] = df['winter']*df['dayuvIndex']
    df['winter_daywindSpeed'] = df['winter']*df['daywindSpeed']
    df['winter_dayHumidity'] = df['winter']*df['dayHumidity']
    df['winter_daytemp'] = df['winter']*df['temp']
    df['winter_daytemp_1'] = df['winter']*df['temp_1']
    df['winter_daytemp_2'] = df['winter']*df['temp_2']
    df['winter_daytemp_7'] = df['winter']*df['temp_7']
    df['winter_daytemp_diff_1'] = df['winter']*df['temp_diff_1']
    df['winter_daytemp_diff_2'] = df['winter']*df['temp_diff_2']
    df['winter_daytemp_diff_7'] = df['winter']*df['temp_diff_7']
        
    df['deepWinter_dayHumidity'] = df['deepWinter']*df['dayHumidity']
    df['deepWinter_dayuvIndex'] = df['deepWinter']*df['dayuvIndex']
    df['deepWinter_daywindSpeed'] = df['deepWinter']*df['daywindSpeed']
    df['deepWinter_dayHumidity'] = df['deepWinter']*df['dayHumidity']
    df['deepWinter_daytemp'] = df['deepWinter']*df['temp']
    df['deepWinter_daytemp_1'] = df['deepWinter']*df['temp_1']
    df['deepWinter_daytemp_2'] = df['deepWinter']*df['temp_2']
    df['deepWinter_daytemp_7'] = df['deepWinter']*df['temp_7']
    df['deepWinter_daytemp_diff_1'] = df['deepWinter']*df['temp_diff_1']
    df['deepWinter_daytemp_diff_2'] = df['deepWinter']*df['temp_diff_2']
    df['deepWinter_daytemp_diff_7'] = df['deepWinter']*df['temp_diff_7']
    
    df['winter_temp_min'] = df['winter']*df['temp_min'] 
    df['winter_temp_min_1'] = df['winter']*df['temp_min_1'] 
    df['deepWinter_temp_min'] = df['deepWinter']*df['temp_min'] 
    df['deepWinter_temp_min_1'] = df['deepWinter']*df['temp_min_1'] 
    
    df['summer_dayHumidity'] = df['summer']*df['dayHumidity']
    
    #pm_train = encoding(pm_train)
    df=df.interpolate()
    df=df.interpolate(method='backfill')
    
    df.loc[df['aqi']<10,'aqi'] = 10 # <1 to 10 was 0.1 better in pm2.5 than <1 to 1, <10 to 10 is 0.05,0.4 better than previos
    df.loc[((df['aqi']>450)&(df['winter']!=1)),'aqi'] = 450
    df['l_aqi'] = np.log(df['aqi'])
    
    df = df[df['type']==worktype]
    
    if if_log:
        Y = df.loc[:,'l_aqi']
    else:
        Y = df.loc[:,'aqi']
    
    X = df.drop(['ID','year','date','latitude','longitude','aqi','l_aqi','dayofmonth','day','source'],axis=1)
    X = X.drop(['winter','summer','type'],axis=1)
    
    
    return Y, X

def encoding(df):
    df = pd.get_dummies(df, columns=['month'], prefix='month', drop_first=True)
    df = pd.get_dummies(df, columns=['hour'], prefix='hour', drop_first=True)
    df = pd.get_dummies(df, columns=['dayofweek'], prefix='dayofweek', drop_first=True)
    df = pd.get_dummies(df, columns=['station'], prefix='stat', drop_first=True)
#    df = pd.get_dummies(df, columns=['type'], prefix='type', drop_first=True)
#    df = pd.get_dummies(df, columns=['source'], prefix='source', drop_first=True)
    return df


def initial_process(df):
    df['date'] = pd.to_datetime(df['date'])
    df['day'] = df['date'].dt.strftime('%Y-%m-%d')
    df['year'] = pd.DatetimeIndex(df['date']).year
    df['month'] = pd.DatetimeIndex(df['date']).month
    df['hour'] = pd.DatetimeIndex(df['date']).hour
    df['dayofweek'] = pd.DatetimeIndex(df['date']).dayofweek
    df['dayofmonth'] = pd.DatetimeIndex(df['date']).day
    return df

def process_weather(weather,temp_var):
    weather['date'] = pd.to_datetime(weather['date'])
    weather['day'] = weather['date'].dt.strftime('%Y-%m-%d')
    weather['year'] = pd.DatetimeIndex(weather['date']).year
    weather['month'] = pd.DatetimeIndex(weather['date']).month
    weather = pd.get_dummies(weather, columns=['icon'], prefix='icon', drop_first=True)
    weather = weather.drop(['Unnamed: 0', 'summary'],axis=1)
    weather = weather.interpolate()
    weather = weather.sort_values(by='date')
    
    whelp =pd.DataFrame()
    whelp['temp'] = weather.groupby(weather['day'])[temp_var].mean()
    whelp['temp_1'] = weather.groupby(weather['day'])[temp_var].mean().shift()
    whelp['temp_2'] = weather.groupby(weather['day'])[temp_var].mean().shift(2)
    whelp['temp_7'] = weather.groupby(weather['day'])[temp_var].mean().shift(7)
    whelp['temp_diff_1'] = weather.groupby(weather['day'])[temp_var].mean().diff()
    whelp['temp_diff_2'] = weather.groupby(weather['day'])[temp_var].mean().diff(2)
    whelp['temp_diff_7'] = weather.groupby(weather['day'])[temp_var].mean().diff(7)
    whelp['temp_min'] = weather.groupby('day')['temperature'].min()
    whelp['temp_min_1'] = weather.groupby('day')['temperature'].min().shift()
    whelp['dayHumidity'] = weather.groupby(weather['day'])['humidity'].mean()
    whelp['dayuvIndex'] = weather.groupby(weather['day'])['uvIndex'].mean()
    whelp['daywindSpeed'] = weather.groupby(weather['day'])['windSpeed'].mean()
    
    whelpmonth = weather.groupby(['year','month'])[temp_var].mean().reset_index()
    whelpmonth = whelpmonth.rename(columns={temp_var: "temp_month_avg"})
        
    weather = weather.merge(whelp,on='day',how='left')
    weather = weather.merge(whelpmonth,on=['year','month'],how='left')
    
    weather = weather.interpolate(method='backfill')
    weather = weather.drop(['day','year','month','dewPoint','precipProbability','precipIntensity'],axis=1)    
    if temp_var == 'apparentTemperature':
        weather = weather.drop(['temperature'],axis=1) 
    else:
        weather = weather.drop(['apparentTemperature'],axis=1) 
    
    lagvars = [temp_var,'windSpeed','windBearing','humidity','uvIndex','visibility']    
    for i in range(1,25):
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
    model = LinearRegression()
    model.fit(X, Y)
#    for i in range(len(X.columns)):
#        print(X.columns[i],model.coef_[i])
    return model
    
