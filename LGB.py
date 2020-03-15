# -*- coding:utf-8 -*-
import time
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
def encode_count(df,column_name):
    lbl = preprocessing.LabelEncoder()
    lbl.fit(list(df[column_name].values))
    df[column_name] = lbl.transform(list(df[column_name].values))
    return df
import warnings
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
warnings.filterwarnings("ignore")
plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文字体设置-黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
sns.set(font='SimHei',font_scale=1.5)  # 解决Seaborn中文显示问题并调整字体大小


train = pd.read_csv('C:/Users/Admin/Desktop/data/train.csv')
test = pd.read_csv('C:/Users/Admin/Desktop/data/test.csv')
train_label = pd.read_csv('C:/Users/Admin/Desktop/data/train_label.csv')
print(train.columns)
feature = [a for a in train.columns if a not in drop_cols]
train = train[feature]
test = test[feature]
train = train.merge(train_label,how='left',on='ID')


cate_feats = []
def display_importances(feature_importance_df_):
    cols = feature_importance_df_[["feature", "importance"]].groupby("feature").mean().sort_values(by="importance",
                                                                                                   ascending=False)[
           :40].index
    best_features = feature_importance_df_.loc[feature_importance_df_.feature.isin(cols)]
    plt.figure(figsize=(8, 10))
    sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False))
    plt.title('LightGBM Features (avg over folds)')
    plt.tight_layout()
    plt.show()




# para
params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    # 'num_class':2,
    'metric': {'auc'},
    'max_depth': -1,
    'learning_rate': 0.01,
    'lambda_l1': 0.1,
    'lambda_l2': 5,  # 越小l2正则程度越高
    'num_leaves':31,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.9,
    'bagging_freq': 2

}

NFOLDS = 10
train_label = train['Label']
kfold = KFold(n_splits=NFOLDS, shuffle=True, random_state=2019)
kf = kfold.split(train, train_label)

train_data_use = train.drop(['ID', 'Label'], axis=1)
test_data_use = test.drop(['ID'], axis=1)

cv_pred = np.zeros(test.shape[0])
valid_best_l2_all = 0

feature_importance_df = pd.DataFrame()
count = 0
for train_index, test_index in kf:
    print('++++++++++++++++fold training+++++++++++++',count)
    X_train, X_validate, label_train, label_validate = \
        train_data_use.loc[train_index], train_data_use.loc[test_index], \
        train_label.loc[train_index], train_label[test_index]
    dtrain = lgb.Dataset(X_train, label_train)
    dvalid = lgb.Dataset(X_validate, label_validate, reference=dtrain)

    bst = lgb.train(params, dtrain, num_boost_round=10000, valid_sets=dvalid, verbose_eval=100, early_stopping_rounds=50)
    cv_pred += bst.predict(test_data_use, num_iteration=bst.best_iteration)


    valid_best_l2_all += bst.best_score['valid_0']['auc']

    fold_importance_df = pd.DataFrame()
    fold_importance_df["feature"] = list(X_train.columns)
    fold_importance_df["importance"] = bst.feature_importance(importance_type='gain', iteration=bst.best_iteration)
    fold_importance_df["fold"] = count + 1
    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
    count += 1


cv_pred /= NFOLDS
valid_best_l2_all /= NFOLDS
print('cv score for valid is: ',  valid_best_l2_all)

display_importances(feature_importance_df)

test_data_sub = test[['ID']]
test_data_sub['Label'] = cv_pred

test_data_sub.to_csv('../result/n_fold_lgb.csv', index=False)

feature_importance_df = feature_importance_df[["feature", "importance"]].groupby(
    "feature").mean().sort_values(by="importance",ascending=False).reset_index()


feature_importance_df.to_csv('../importance/importance.csv', index=False)