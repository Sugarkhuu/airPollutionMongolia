lin_weight = 0.5
cat_weight = 0.5

linear = pd.read_csv('sub_linear.csv')
cat    = pd.read_csv('sub_cat.csv')

my_sub = linear.copy()
my_sub['aqi'] = lin_weight*linear['aqi'] + cat_weight*cat['aqi']



subB = pd.read_csv('sub605.csv')
print(np.sqrt(mean_squared_error(subB['aqi'],my_sub['aqi'])))


plt.scatter(subB['aqi'],my_sub['aqi'])
subB['aqi'].plot()
my_sub['aqi'].plot()

subB['aqi'].hist(bins=100)
my_sub['aqi'].hist(bins=100)



submission = my_sub[['ID','aqi']].copy()
assert submission['aqi'].isnull().sum() == 0
submission.to_csv('submission.csv',index=False)