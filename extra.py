import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import os

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

#pm_train_c['y_test'].to_csv("xgboost_train.csv",index=False)
#pm_train_c['y_test'].to_csv("catboost_train.csv",index=False)
#pm_train_c['y_test'].to_csv("linear_train.csv",index=False)

subB = pd.read_csv('sub601.csv')

plt.scatter(subB['aqi'],pm_test['aqi'])
subB['aqi'].plot()
pm_test['aqi'].plot()

subB['aqi'].hist(bins=100)
pm_test['aqi'].hist(bins=100)


plt.scatter(sub_nan['aqi'],pm_test['aqi'])
sub_nan['aqi'].plot()
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


tmp = pm_train.groupby(['station','month','hour'])['aqi'].mean()
tmp=tmp.reset_index()
tmp = tmp.rename(columns={'aqi': "aqi_mean"})
pm_test = pm_test.merge(tmp,on=['station','month','hour'],how='left')

pm_test.error.hist(bins=10)

plt.scatter(pm_test['aqi_mean'],pm_test['aqi'])

pm_test['aqi'].hist(bins=100)
pm_test['aqi_mean'].hist(bins=100)

lin  = pd.read_csv('submission_linear.csv')

pm_test['aqi_mean'].plot()
pm_test['aqi'].plot()
lin['aqi'].plot()



pm_test.iloc[10000:,:].error.hist(bins=10)

cat = pd.read_csv('catboost_train.csv')
xg  = pd.read_csv('xgboost_train.csv')
lin  = pd.read_csv('linear_train.csv')

pm_train = pm_train_c.copy()
pm_train['cat_hat'] = cat['y_test'].values
pm_train['xg_hat'] = xg['y_test'].values
pm_train['lin_hat'] = lin['y_test'].values
pm_train['cat_err'] = pm_train['cat_hat'] - pm_train['aqi']
pm_train['xg_err'] = pm_train['xg_hat'] - pm_train['aqi']
pm_train['lin_err'] = pm_train['lin_hat'] - pm_train['aqi']

pm_train = pm_train.merge(weather,on='date',how='left')
#err200 = train[train['error']<-200]

months = [1,2,3];[10,11,12]
#day = 25
year = 2018
station = 5
ntype = 1

sample = pm_train[((pm_train['month'].isin(months))&(pm_train['type']==types[ntype])&(pm_train['station']==stations[station])&(pm_train['year']==year))]
sample[['aqi','lin_hat']].plot()
sample[['windSpeed','apparentTemperature']].plot()
sample[['y_test']].plot()

#pm_test = pm_test.merge(weather,on='date',how='left')
#
#import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
#
#
#X = pm_train['apparentTemperature']
#Y = pm_train['aqi']
#Z = pm_train['windSpeed']
#
#fig = plt.figure(figsize = (10, 7))
#ax = plt.axes(projection ="3d")
##ax.scatter3D(draw['hour'], draw['aqi'], draw['temperature'], c= draw['dayofmonth'])
#ax.scatter3D(X, Y, Z)
#plt.show()

db = pm_train_c
x_var = 'month'
y_var = 'aqi'
type_var = 'dayofweek';'hour';'station';'year';'type';'station'
col_var = 'year'
an_var = 'hour'
var2c = 'aqi'

plt.figure()
sns.boxplot(y=y_var, x=x_var, 
                 data=db, 
                 palette="colorblind",
                 hue=type_var)

plt.figure()
sns.boxplot(y='cat_err', x=x_var, 
                 data=pm_train, 
                 palette="colorblind",
                 hue='type')

plt.figure()
sns.boxplot(y='xg_err', x=x_var, 
                 data=pm_train, 
                 palette="colorblind",
                 hue='type')





g = sns.FacetGrid(db, row=x_var, col=col_var,margin_titles=True)
g.map(sns.regplot, an_var, var2c, color=".3", fit_reg=False, x_jitter=.1)


g = sns.FacetGrid(pm_train, row="year", col="month",margin_titles=True)
g.map(sns.regplot, "hour", "cat_err", color=".3", fit_reg=False, x_jitter=.1)
g = sns.FacetGrid(pm_train, row="year", col="month",margin_titles=True)
g.map(sns.regplot, "hour", "lin_err", color=".3", fit_reg=False, x_jitter=.1)


print(np.sqrt(mean_squared_error(subB['aqi'],pm_test['aqi'])))
print(np.sqrt(mean_squared_error(subB['aqi'],pm_test['aqi'])))
print(np.sqrt(mean_squared_error(subB['aqi'],pm_test['aqi'])))


test_cat = pd.read_csv('sub_cat.csv')
test_xg = pd.read_csv('sub_xg.csv')
test_lin = pd.read_csv('sub_lin.csv')
test_best = pd.read_csv('sub606.csv')


test_lin['aqi'].plot()


test_cat['aqi'].plot()

test_best['aqi'].plot()
test_xg['aqi'].plot()

linAndCat = test_lin.copy()
linAndCat['aqi'] = 0.5*test_lin['aqi'] + 0.5*test_xg['aqi']

plt.scatter(test_best['aqi'],linAndCat['aqi'])

test_best['aqi'].hist(bins=100)
linAndCat['aqi'].hist(bins=100)

test_best['aqi'].plot()
linAndCat['aqi'].plot()


print(np.sqrt(mean_squared_error(test_best['aqi'],linAndCat['aqi'])))


diff = pd.DataFrame()
diff['lin_cat'] = test_lin['aqi'] - test_cat['aqi']
diff['lin_xg'] = test_lin['aqi'] - test_xg['aqi']
diff['xg_cat'] = test_xg['aqi'] - test_cat['aqi']



diff['lin_xg'].plot()
diff['xg_cat'].plot()
plt.title('blue: lin-xg,yellow: xg-cat')

diff['lin_cat'].plot()
diff['xg_cat'].plot()
plt.title('blue: lin-cat,yellow: xg-cat')

diff['lin_cat'].plot()
diff['lin_xg'].plot()
plt.title('blue: lin-cat,yellow: lin-xg')

fig = plt.figure()
plt.subplot(311)
diff['lin_cat'].plot()
plt.subplot(312)
diff['lin_cat'].plot()
plt.subplot(313)
diff['lin_cat'].plot()



plt.scatter(diff['lin_cat'],diff['lin_xg'])


pm_test.groupby(['year','month'])['error'].mean()
pm_test.groupby(['year','month'])['error'].std()







X=X_train.copy()
model=my_model
for i in range(len(X.columns)):
    print(X.columns[i],model.coef_[i])
    
















