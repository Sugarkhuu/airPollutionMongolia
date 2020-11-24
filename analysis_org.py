import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os


sugar_dir = '/home/sugarkhuu/Documents/python/airPollutionMongolia'
os.chdir(sugar_dir)


pm_train = pd.read_csv('./ulaanbaatar-city-air-pollution-prediction/pm_train.csv')
pm_test = pd.read_csv('./ulaanbaatar-city-air-pollution-prediction/pm_test.csv')
weather = pd.read_csv('./ulaanbaatar-city-air-pollution-prediction/weather.csv')

pm_train.head()
# stations
len(pm_train['station'].unique()), len(pm_test['station'].unique())


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
    
    
# Тестийн датаны хувьд датаны мөрийн тоо (сараар)
time_series_graph(test_count)

# Цаг агаарын датаны хувьд датаны мөрийн тоо (сараар)
time_series_graph(weather_count)


# Хотод байгаа бүх станцын дунджаар дата бэлдэх.
def get_records(df):
    data = df[['date' , 'type', 'aqi']]
    data = data.fillna(0)
    pv = pd.pivot_table(data, values='aqi', index=['date'], columns=['type'], aggfunc=np.mean)
    records = pd.DataFrame(pv.to_records())
    return records

train_records = get_records(pm_train)
test_records = get_records(pm_test)
test_records.info()


temp = weather[['date', 'temperature']]
temp = temp.set_index('date')
temp.plot(figsize=(15,8), rot=90)

# Зарим цагуудын дата дутуу тул дутуу цагийн мөрийг оруулж ирэх
def fill_date(df):
    ranges = pd.date_range(df['date'].min(),df['date'].max(), freq='H')
    filled = pd.DataFrame(ranges, columns= ['date']).merge(df, on=['date'], how='left')
    return filled

train_records_filled = fill_date( train_records)
test_records_filled = fill_date( test_records)

train = pd.merge(train_records_filled, temp, on=['date'], how='left')
test = pd.merge(test_records_filled, temp, on=['date'], how='left')

train[train['PM10'].isna()]

train_inter = train.interpolate()
test_inter = test.interpolate()
print(train_inter[train_inter['PM10'].isna()])
train_inter.head()

print(test_inter[test_inter['PM10'].isna()])


#Хэд дэх сар, өдөр мөн цаг маань тоосонцрын хэмжээнд нөлөөлдөг байх боломжтой гэж үзээд date features бэлдэх
def set_date_features(df):
    df['month'] = pd.DatetimeIndex( df['date']).month
    df['dayofweek'] = pd.DatetimeIndex( df['date']).dayofweek
    df['hour'] = pd.DatetimeIndex( df['date']).hour
    return df

train_data = set_date_features(train_inter)
test_data = set_date_features(test_inter)

columns = ['date','PM10', 'PM2.5' , 'temperature','month','dayofweek','hour']
train = train_data[columns]
test = test_data[columns]


# onehot encoding
def encoding(df):
    df = df.drop(['date'], axis = 1) 
    df = pd.get_dummies(df, columns=['month'], prefix='month')
    df = pd.get_dummies(df, columns=['hour'], prefix='hour')
    df = pd.get_dummies(df, columns=['dayofweek'], prefix='dayofweek')
    return df


train_encoded = encoding(train)
test_encoded = encoding(test)
# 2 years data
n_train_hours = int( 365 * 24 * 2 )

train_data = train_encoded.values[:n_train_hours, :]
valid_data = train_encoded.values[n_train_hours:, :]
test_data = test_encoded.values

train_X, train_y = train_data[:, 2:], train_data[:, 0:2]
valid_X, valid_y = valid_data[:, 2:], valid_data[:, 0:2]
test_X, test_y = test_data[:, 2:], test_data[:, 0:2]

dates = test['date'].values


### Энгийн linear model турших¶

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


model = RandomForestRegressor(n_estimators=100,
                              max_depth=3,
                              random_state=10)
model.fit(train_X, train_y) 
y_hat= model.predict(valid_X)

print('MSE On validation Data: {}'.format( np.sqrt(mean_squared_error(valid_y,y_hat))))

y_hat= model.predict(test_X)

dates = dates.reshape(len(dates),1)

sub = np.concatenate((dates.astype(str), y_hat.astype(str)),axis=1)
tmp = pd.DataFrame(sub, columns=['date', 'PM10','PM2.5'])

tmp['date']= pd.to_datetime(tmp['date'])
pm25 = pm_test[pm_test['type']=='PM2.5']
pm10 = pm_test[pm_test['type']=='PM10']

sub_pm25 = pd.merge(pm25, tmp, on=['date'], how='left')
sub_pm10 = pd.merge(pm10, tmp, on=['date'], how='left')

submission = pd.DataFrame(columns=['ID', 'aqi'])
submission = submission.append(sub_pm25[['ID', 'PM2.5']].rename(columns={'PM2.5': 'aqi'}))
submission = submission.append(sub_pm10[['ID', 'PM10']].rename(columns={'PM10': 'aqi'}))






