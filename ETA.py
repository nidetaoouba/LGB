#!/usr/bin/python
# -*- coding:utf8 -*-tf-8
import pandas as pd
import numpy as np
import lightgbm as lgb
# Plots
import seaborn as sns
import matplotlib.pyplot as plt


# Data processing, metrics and modeling
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import precision_score, recall_score, confusion_matrix,  roc_curve, precision_recall_curve, accuracy_score, roc_auc_score
import lightgbm as lgbm

# Stats
import scipy.stats as ss
from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform

# Time
from contextlib import contextmanager
@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.0f}s".format(title, time.time() - t0))

#ignore warning messages
import warnings
warnings.filterwarnings('ignore')

#读取文件
train1 = pd.read_csv('C:/Users/Admin/Desktop/data/new data/train0523 2/train0523.csv')
train2 = pd.read_csv('C:/Users/Admin/Desktop/data/new data/event_port/loadingOrderEvent.csv')
train3 = pd.read_csv('C:/Users/Admin/Desktop/data/new data/event_port/port.csv')
print(train1.head(10))
#train_result = pd.read_csv('C:/Users/Admin/Desktop/data/train_label.csv')
test = pd.read_csv('C:/Users/Admin/Desktop/data/new data/A_testData0531.csv')
#print(test.head(10))
#display(train.head(10),test.head(10),train_result.head())
train3 = train3.rename(columns={'TRANS_NODE_NAME':'EVENT_LOCATION_ID'},inplace=True)



train1.shape  #训练样本总数

test.shape  #测试样本总数




#train['Type'] = 'train'
#test['Type'] = 'test'
#data1 = train.append(test)
data1 = train1.merge(train2,on='loadingOrder',how='left')
data = data1.merge(train3,on='EVENT_LOCATION_ID',how='left')
#data.drop(['经营期限至', '邮政编码', '核准日期', '注销时间', '经营期限自', '成立日期', '经营范围'],inplace=True,axis=1)
data['Type'] = 'train'
test['Type'] = 'test'
data = data.append(test)


data.shape


# In[230]:


#display(data.head(10))



# In[231]:


def display_missing_ratio(data):
        data_na = (data.isnull().sum() / len(data)) * 100
        data_na = data_na.sort_values(ascending=False)
        # data_na = data_na.drop(data_na[data_na == 0].index).sort_values(ascending=False)[:30]
        missing_data = pd.DataFrame({'Missing Ratio': data_na})
        print(missing_data)




train = data[data['Type'] == 'train']
test = data[data['Type'] == 'test']

train = train.drop(columns = ['Type'])
test = test.drop(columns = ['Type'])



X = train.drop('Label', 1)
y = train['Label']
X_test = test
X_test = X_test.drop(columns = ['Label'])


# In[234]:


random_state = 42
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size = 0.2, random_state = random_state)

print(X_train.shape,X_valid.shape)


# In[ ]:


fit_params = {"early_stopping_rounds" : 100,
             "eval_metric" : 'auc',
             "eval_set" : [(X_train,y_train)],
             'eval_names': ['valid'],
             'verbose': 0,
             'categorical_feature': 'auto'}

param_test = {'learning_rate' : [0.01, 0.02, 0.03, 0.04, 0.05, 0.08, 0.1, 0.2, 0.3, 0.4],
              'n_estimators' : [100, 200, 300, 400, 500, 600, 800, 1000, 1500, 2000],
              'num_leaves': sp_randint(6, 50),
              'min_child_samples': sp_randint(100, 500),
              'min_child_weight': [1e-5, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4],
              'subsample': sp_uniform(loc=0.2, scale=0.8),
              'max_depth': [-1, 1, 2, 3, 4, 5, 6, 7],
              'colsample_bytree': sp_uniform(loc=0.4, scale=0.6),
              'reg_alpha': [0, 1e-1, 1, 2, 5, 7, 10, 50, 100],
              'reg_lambda': [0, 1e-1, 1, 5, 10, 20, 50, 100]}

#number of combinations
n_iter = 500

#intializing lgbm and lunching the search
lgbm_clf = lgbm.LGBMClassifier(random_state=random_state, silent=True, metric='None', n_jobs=4)
grid_search = RandomizedSearchCV(
    estimator=lgbm_clf, param_distributions=param_test,
    n_iter=n_iter,
    scoring='accuracy',
    cv=5,
    refit=True,
    random_state=random_state,
    verbose=True)

grid_search.fit(X, y, **fit_params)
print('Best score reached: {} with params: {} '.format(grid_search.best_score_, grid_search.best_params_))

opt_parameters =  grid_search.best_params_


# In[235]:


lgbm_clf = lgbm.LGBMClassifier(**opt_parameters)
lgbm_clf.fit(X, y)
y_pred = lgbm_clf.predict(X_test)


# In[222]:


temp = pd.DataFrame(pd.read_csv("C:/Users/BM/Desktop/data/test.csv")['ID'])
temp['Label'] = pred
temp.to_csv("C:/Users/Admin/Desktop/data/submission01.csv", index = False)

