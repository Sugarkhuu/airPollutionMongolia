lin_weight = 0.50
cat_weight = 0.50
xg_weight  = 0.00

linear = pd.read_csv('submission_linear.csv')
cat    = pd.read_csv('submission_cat.csv')
#linear = pd.read_csv('sub_linear.csv')
#cat    = pd.read_csv('sub_cat.csv')
#xg    = pd.read_csv('sub_xg.csv')


my_sub = linear.copy()
my_sub['aqi'] = lin_weight*linear['aqi'] + cat_weight*cat['aqi']
#my_sub['aqi'] = lin_weight*linear['aqi'] + cat_weight*cat['aqi'] + xg_weight*xg['aqi']


tmp = pm_train.groupby(['station','month','hour'])['aqi'].mean()
tmp=tmp.reset_index()
tmp = tmp.rename(columns={'aqi': "aqi_mean"})
#del pm_test['aqi_mean_y']
#pm_test = pm_test.rename(columns={'aqi_mean_x': "aqi_mean"})
pm_test = pm_test.merge(tmp,on=['station','month','hour'],how='left')


check = my_sub.copy()
check['aqi_mean_diff'] = my_sub['aqi'] - pm_test['aqi_mean']
check['aqi_mean_diff'].hist(bins=100)



corr_factor = 0.5;.75
corr_thres  =  50;25
#my_sub.loc[check['aqi_mean_diff']<-100,'aqi'] = my_sub.loc[check['aqi_mean_diff']<-100,'aqi'] + 15
my_sub.loc[check['aqi_mean_diff']>corr_thres,'aqi'] = my_sub.loc[check['aqi_mean_diff']>corr_thres,'aqi'] - corr_factor*check.loc[check['aqi_mean_diff']>corr_thres,'aqi_mean_diff']

# readjust for those already given
my_sub.loc[~pm_test['error'].isnull(),'aqi'] = linear.loc[~pm_test['error'].isnull(),'aqi']



submission = my_sub[['ID','aqi']].copy()
assert submission['aqi'].isnull().sum() == 0
submission.to_csv('submission.csv',index=False)


###############################################################################

subB = pd.read_csv('sub6002.csv')
print(np.sqrt(mean_squared_error(subB['aqi'],my_sub['aqi'])))


plt.scatter(subB['aqi'],my_sub['aqi'])

subB['aqi'].plot()
my_sub['aqi'].plot()

linear['aqi'].plot()
cat['aqi'].plot()

subB['aqi'].hist(bins=100)
my_sub['aqi'].hist(bins=100)


subB['aqi'].plot()
pm_test['aqi_mean'].plot()

subB['aqi'].plot()
pm_test['aqi'].plot()

subB['aqi'].hist(bins=100)
pm_test['aqi'].hist(bins=100)