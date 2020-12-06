lin_weight = 0.5
cat_weight = 0.5

linear = pd.read_csv('submission_linear.csv')
cat    = pd.read_csv('submission_cat.csv')

my_sub = linear.copy()
my_sub['aqi'] = lin_weight*linear['aqi'] + cat_weight*cat['aqi']


check = my_sub.copy()
check['aqi_mean_diff'] = my_sub['aqi'] - pm_test['aqi_mean']
check['aqi_mean_diff'].hist(bins=100)



#my_sub.loc[check['aqi_mean_diff']<-100,'aqi'] = my_sub.loc[check['aqi_mean_diff']<-100,'aqi'] + 15
my_sub.loc[check['aqi_mean_diff']>50,'aqi'] = my_sub.loc[check['aqi_mean_diff']>50,'aqi'] - 25


submission = my_sub[['ID','aqi']].copy()
assert submission['aqi'].isnull().sum() == 0
submission.to_csv('submission.csv',index=False)


###############################################################################

subB = pd.read_csv('sub605.csv')
print(np.sqrt(mean_squared_error(subB['aqi'],my_sub['aqi'])))


plt.scatter(subB['aqi'],my_sub['aqi'])
subB['aqi'].plot()
my_sub['aqi'].plot()

subB['aqi'].hist(bins=100)
my_sub['aqi'].hist(bins=100)


subB['aqi'].plot()
pm_test['aqi_mean'].plot()






