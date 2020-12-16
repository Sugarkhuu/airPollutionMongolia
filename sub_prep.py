# 1: cat - (lin-cat) 1
# 2: lin - (cat-lin) 29089
# 3: average 76242

lin_weight = 0.50
cat_weight = 0.50
#xg_weight  = 0.00

#linear = pd.read_csv('submission_linear.csv')
#cat    = pd.read_csv('submission_cat.csv')
linear = pd.read_csv('sub_linear.csv')
cat    = pd.read_csv('sub_cat.csv')
#xg    = pd.read_csv('sub_xg.csv')




my_sub = linear.copy()
my_sub['aqi'] = lin_weight*linear['aqi'] + cat_weight*cat['aqi']


pm_test1 = pm_test.iloc[:29089]
pm_test2 = pm_test.iloc[29089:76242]

pm_test1 = pm_test1.merge(b1,on='hour',how='left')
pm_test2 = pm_test2.merge(b2,on='hour',how='left')

my_sub['shift'] = 0
my_sub.loc[:29088,'shift'] = pm_test1['shift'].values
my_sub.loc[29089:76241,'shift'] = pm_test2['shift'].values
my_sub.loc[my_sub['shift']<0,'shift'] = 0


my_sub['aqi'] = my_sub['aqi'] + my_sub['shift']/4
#my_sub.loc[my_sub['aqi']<10,'aqi'] = my_sub.loc[my_sub['aqi']<10,'aqi'] + my_sub.loc[my_sub['aqi']<10,'shift']/4

#my_sub.loc[:29088,'aqi'] = my_sub.loc[:29088,'aqi'] - 40
#my_sub.loc[29089:76241,'aqi'] = my_sub.loc[29089:76241,'aqi'] - 20
#my_sub.loc[76242:,'aqi'] = my_sub.loc[76242:,'aqi']

#my_sub.loc[:29088,'aqi'] = cat.loc[:29088,'aqi']-(linear.loc[:29088,'aqi']-cat.loc[:29088,'aqi'])
#my_sub.loc[29089:76241,'aqi'] = linear.loc[29089:76241,'aqi']-(cat.loc[29089:76241,'aqi']-linear.loc[29089:76241,'aqi'])
#my_sub.loc[76242:,'aqi'] = lin_weight*linear.loc[76242:,'aqi'] + cat_weight*cat.loc[76242:,'aqi']

#my_sub['aqi'] = lin_weight*linear['aqi'] + cat_weight*cat['aqi'] + xg_weight*xg['aqi']


tmp = pm_train.groupby(['station','month','hour'])['aqi'].mean()
tmp=tmp.reset_index()
tmp = tmp.rename(columns={'aqi': "aqi_mean"})
#del pm_test['aqi_mean_y']
#pm_test = pm_test.rename(columns={'aqi_mean_x': "aqi_mean"})
pm_test = pm_test.merge(tmp,on=['station','month','hour'],how='left')
#
#
check = my_sub.copy()
check['aqi_mean_diff'] = my_sub['aqi'] - pm_test['aqi_mean']
check['aqi_mean_diff'].hist(bins=100)
#
#
#
corr_factor = 0.5;.75
corr_thres  =  25;50;25
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

subB['aqi'].plot()
my_sub['aqi'].plot()


my_sub['aqi'].plot()
linear['aqi'].plot()
cat['aqi'].plot()

subB['aqi'].hist(bins=100)
my_sub['aqi'].hist(bins=100)


subB['aqi'].plot()
pm_test['aqi_mean'].plot()



subB = pd.read_csv('sub6002.csv')
sub_xg_old = pd.read_csv('sub_xg.csv')
sub_xg_new = pd.read_csv('submission.csv')




sub_xg_old['aqi'].plot()
sub_xg_new['aqi'].plot()

sub_xg_old['aqi'].hist(bins=100)
sub_xg_new['aqi'].hist(bins=100)

subB['aqi'].plot()
sub_xg_new['aqi'].plot()

subB['aqi'].hist(bins=100)
sub_xg_new['aqi'].hist(bins=100)

submix = subB.copy()
submix['aqi'] = (subB['aqi'] + sub_xg_new['aqi'])/2

submission = submix[['ID','aqi']].copy()
assert submission['aqi'].isnull().sum() == 0
submission.to_csv('submission_mix.csv',index=False)
