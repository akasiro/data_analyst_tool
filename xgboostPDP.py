import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
# from sklearn.ensemble import GradientBoostingRegressor
from sklearn.inspection import PartialDependenceDisplay
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
def xgboostPDP(data,x,y,**kwargs):
    ''' 
    Args:
        data: (pandas.DataFrame)
        x (list)
        y (list)
        **kwargs : args for XGBRegressor, using "??XGBRegressor" for more info
    returns:
        data,clf,featureImportance,fig,fig_feature
    '''    
    data = data[x+y].dropna(how='any')
    print(data.shape)
    clf = XGBRegressor(**kwargs).fit(data[x],data[y])
    data['y_pred'] = clf.predict(data[x])
    print('r2: {:.6f}'.format(r2_score(data[y],data['y_pred'])))
    print('mae: {:.6f}'.format(mean_absolute_error(data[y],data['y_pred'])))
    print('mse: {:.6f}'.format(mean_squared_error(data[y],data['y_pred'])))
    featureImportance = pd.DataFrame({'feature':x,'feature_importance':clf.feature_importances_}).sort_values('feature_importance',ascending=False).reset_index(drop=True)
    features = featureImportance['feature'].values.tolist()
    print(featureImportance)

    display = PartialDependenceDisplay.from_estimator(clf, data[x],features=features)
    fig = plt.gcf()
    axes = fig.axes[1:]
    for i in range(len(features)):
        ax = axes[i]
        x_min,x_max = ax.get_xlim()
        xscale = ax.get_xscale()
        sns.rugplot(data[features[i]],height=0.05,alpha = 0.005,color='#CF3512',ax=ax)
        ax.set_xlim([x_min,x_max])
        ax.set_xscale(xscale)
    
    fig_feature,ax1 = plt.subplots()
    sns.barplot(y = featureImportance['feature'],x = featureImportance['feature_importance'],ax=ax1)
    for i in ax1.patches:
        _x = i.get_x() + i.get_width()
        _y = i.get_y() + 0.5*i.get_height()
        value = '{:.3f}'.format(i.get_width())
        ax1.text(_x,_y,value,ha='left')
    plt.show()
    return data,clf,featureImportance,fig,fig_feature