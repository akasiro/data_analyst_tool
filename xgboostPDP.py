import math
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from xgboost import XGBClassifier
# from sklearn.ensemble import GradientBoostingRegressor
from sklearn.inspection import PartialDependenceDisplay
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error,roc_auc_score,roc_curve
from sklearn.model_selection import train_test_split as TTS
def xgboostPDP(data,x,y,type='r',**kwargs):
    ''' 
    Args:
        data: (pandas.DataFrame)
        x (list)
        y (list)
        type (string): r or c
        **kwargs : args for XGBRegressor, using "??XGBRegressor" for more info
    returns:
        data,clf,featureImportance,fig,fig_feature
    '''    
    data = data[x+y].dropna(how='any')
    
    print(data.shape)
    test,train = TTS(data,test_size=0.3)
    if type == 'r':
        clf = XGBRegressor(**kwargs).fit(train[x],train[y])
    elif type == 'c':
        clf = XGBClassifier(**kwargs).fit(train[x],train[y])
    test['y_pred'] = clf.predict(test[x])
    print('r2: {:.6f}'.format(r2_score(test[y],test['y_pred'])))
    print('mae: {:.6f}'.format(mean_absolute_error(test[y],test['y_pred'])))
    print('mse: {:.6f}'.format(mean_squared_error(test[y],test['y_pred'])))
    if type == 'c':
        print('auc: {:.6f}'.format(roc_auc_score(test[y],test['y_pred'])))
    featureImportance = pd.DataFrame({'feature':x,'feature_importance':clf.feature_importances_}).sort_values('feature_importance',ascending=False).reset_index(drop=True)
    features = featureImportance['feature'].values.tolist()
    print(featureImportance)

    display = PartialDependenceDisplay.from_estimator(clf, test[x],features=features)
    fig = plt.gcf()
    fig.set_figheight(fig.get_figheight()*(math.ceil(len(features)/3)))
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