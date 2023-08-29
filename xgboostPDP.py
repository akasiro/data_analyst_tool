# %load ../tool/daTool/xgboostPDP.py
import math
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from xgboost import XGBClassifier
# from sklearn.ensemble import GradientBoostingRegressor
from sklearn.inspection import partial_dependence #用于计算pdp每个点的值
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error,roc_auc_score,roc_curve
from sklearn.model_selection import train_test_split as TTS
from scipy.interpolate import splev, splrep # 数据平滑插值
def create_subplot_figure(num_plots,fixColsNum = 3):
    '''
    生成一个固定列数的figure
    Args:
        num_plots (int): 总ax数
        fixColsNum (int): 固定行数
    Returns:
        (plt.Figures)
        (plt.axes): axes的序号已经被重置为一位数

    '''
    num_rows = (num_plots + 2) // fixColsNum  # 计算行数
    num_cols = min(num_plots, fixColsNum)  # 每行最多fixColsNum个ax

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(6*num_cols, 6*num_rows))

    # 将axes转换为一维数组，方便后续操作
    axes = axes.flatten()

    # 隐藏多余的子图
    for i in range(num_plots, num_rows*num_cols):
        fig.delaxes(axes[i])

    return fig, axes

def plot_pdp_feature_importance(model,test,x):
    ''' 
    Args:
        model: machine learning model
        test (pandas.DataFrame): 测试集 test dataset
        x (list): feature list, 模型特征
    Returns:
        (pandas.DataFrame): 特征重要性表，已经按重要性倒序
        (plt.Figure): 输出的图
    '''
    featureImportance = pd.DataFrame({'feature':x,'feature_importance':model.feature_importances_}).sort_values('feature_importance',ascending=False).reset_index(drop=True)
    features = featureImportance['feature'].values.tolist()
    print(featureImportance)
    # the number of feature,特征个数
    featuresNum = len(features)
    # create figures with axs,根据特征个数建立图片
    fig, axs = create_subplot_figure(featuresNum+1)
    for i in range(featuresNum):
        pdp = partial_dependence(model,test[x],[features[i]],kind='both',grid_resolution=50) #参数both，表示除了pdp均值外会计算出每个样本的ice
        sns.set_theme(style='ticks',palette='deep',font_scale=1.1)
        plot_x = pd.Series(pdp['values'][0]).rename('x')
        plot_i = pdp['individual'] #获取每个样本的ice数组
        plot_y = pdp['average'][0]
        tck = splrep(plot_x, plot_y, s=30)# 进行数据平滑，参考：https://docs.scipy.org/doc/scipy/reference/tutorial/interpolate.html?highlight=interpolate
        xnew = np.linspace(plot_x.min(),plot_x.max(),300)
        ynew = splev(xnew, tck, der=0)

        plot_df = pd.DataFrame(columns=['x','y']) # 定义一个新df，包括x y两列
        for a in plot_i[0]: # 遍历每个样本的ice
            a2 = pd.Series(a)
            df_i = pd.concat([plot_x, a2.rename('y')], axis=1) # 将ice与x横向拼接
            plot_df = pd.concat([plot_df,df_i], axis= 0) # 加入到总表中

        sns.lineplot(data=plot_df, x="x", y="y", color='k', linewidth = 1.5, linestyle='--', alpha=0.6,ax=axs[i]) # 使用sns绘制线图，如果同一个x上有多个y的话将会自动生成95%置信区间
        axs[i].plot(xnew, ynew, linewidth=2)  # 绘制平滑曲线
        sns.rugplot(data = test.sample(min(200,test.shape[0])), x = features[i], height=.05, color='k', alpha = 0.3,ax = axs[i])# 使用sns绘制轴须图

        x_min = plot_x.min()-(plot_x.max()-plot_x.min())*0.1 #x轴下限
        x_max = plot_x.max()+(plot_x.max()-plot_x.min())*0.1 #x轴上限

        axs[i].set_xlim(x_min,x_max)
        axs[i].set_title(features[i] , fontsize=12)
        axs[i].set_ylabel('Partial Dependence',fontsize = 10)
    
    sns.barplot(y = featureImportance['feature'],x = featureImportance['feature_importance'],ax=axs[featuresNum])
    for i in axs[featuresNum].patches:
        _x = i.get_x() + i.get_width()
        _y = i.get_y() + 0.5*i.get_height()
        value = '{:.3f}'.format(i.get_width())
        axs[featuresNum].text(_x,_y,value,ha='left',fontsize=8)
    axs[featuresNum].set_title('Feature importance',fontsize = 10) 
    return featureImportance,fig

def xgboostPDP(data,x,y,type='r',test_size=0.3,**kwargs):
    ''' 
    函数执行3个步骤:1划分训练集和测试集; 2xgboost建模; 3绘制pdp图
    Args:
        data: (pandas.DataFrame): 数据集
        x (list): feature list, 模型特征
        y (list): 模型标签
        type (string): r or c, 回归模型时使用'r',分类模型使用'c'
        test_size (float): 测试集比例
        **kwargs : args for XGBRegressor or XGBClassifier, using "??XGBRegressor" for more info
    returns:
        (xgboost.XGBRegressor or xgboost.XGBClassifier): xgboost model
        (pandas.DataFrame): the test dataset
        (pandas.DataFrame): the train dataset
        (pandas.DataFrame): 特征重要性表，按重要性倒排
        (matplotlib.pylib.Figure): pdp图
    '''    
    data = data[x+y].dropna(how='any')
    print('The shape of data source: {}'.format(data.shape))
    
    test,train = TTS(data,test_size=test_size)
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
    featureImportance,fig = plot_pdp_feature_importance(clf,test,x)
    return clf,test,train,featureImportance,fig
