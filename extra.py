pm_train = pm_train_c.copy()
pm_train = pm_train.merge(weather,on='date',how='left')
#err200 = train[train['error']<-200]




months = [1,2,3];[10,11,12]
#day = 25;15;14
year = 2018
station = 3;9;8;11;8;1;6
ntype = 0

sample = pm_train[((pm_train['month'].isin(months))&(pm_train['type']==types[ntype])&(pm_train['station']==stations[station])&(pm_train['year']==year))]

sample[['aqi','y_test']].plot()
sample[['windSpeed','apparentTemperature']].plot()
sample[['y_test']].plot()


smpl3 = sample[sample['aqi']>300]




pm_test = pm_test.merge(weather,on='date',how='left')




import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


X = pm_train['apparentTemperature']
Y = pm_train['aqi']
Z = pm_train['windSpeed']

fig = plt.figure(figsize = (10, 7))
ax = plt.axes(projection ="3d")
#ax.scatter3D(draw['hour'], draw['aqi'], draw['temperature'], c= draw['dayofmonth'])
ax.scatter3D(X, Y, Z)
plt.show()


Axes3D.plot_surface(X=X, Y=Y, Z=Z)

