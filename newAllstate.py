#!/usr/bin/python
# -*- coding:utf8 -*-

import numpy as np
import pandas as pd
import xgboost as xgb
import operator
from scipy.stats import norm, lognorm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

# 检查文件是否存在
# from subprocess import check_output
# print(check_output(["ls", "./input"]).decode("utf8"))

# 读取数据
train = pd.read_csv("./input/train.csv")
test = pd.read_csv("./input/test.csv")

# 移除'id' 'loss',并重命名为features
features = [x for x in train.columns if x not in ['id', 'loss']]
# 根据type种类,分离cat和cont
cat_features = [x for x in train.select_dtypes(include=['object']).columns if x not in ['id', 'loss']]
num_features = [x for x in train.select_dtypes(exclude=['object']).columns if x not in ['id', 'loss']]

# 没懂
train['log_loss'] = np.log(train['loss'])
# 提取train和test行数
ntrain = train.shape[0]
ntest = test.shape[0]
# 合并除去id和loss之后的train和test
train_test = pd.concat((train[features], test[features])).reset_index(drop=True)
# 讲cat下的字母转换成category codes
for c in range(len(cat_features)):
    train_test[cat_features[c]] = train_test[cat_features[c]].astype('category').cat.codes

# 分离处理后的train和test的数据
train_x = train_test.iloc[:ntrain, :]
test_x = train_test.iloc[ntrain:, :]
# XGBoost介绍 http://www.10tiao.com/html/284/201608/2652390173/1.html
# 使用XGBoost迭代数据,建立训练集
xgdmat = xgb.DMatrix(train_x, train['log_loss'])
# eta如同学习率 subsample采样训练数据 colsample_bytree构建树树时的采样比率 max_depth构建树的深度 min_child_weight节点的最少特征数
params = {'eta': 0.01, 'seed':0, 'subsample': 0.5, 'colsample_bytree': 0.5, 'objective': 'reg:linear', 'max_depth':6, 'min_child_weight':3}
# 迭代次数
num_rounds = 1000
bst = xgb.train(params, xgdmat, num_boost_round=num_rounds)



def create_feature_map(features):
    outfile = open('xgb.fmap', 'w')
    i = 0
    for feat in features:
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))
        i += 1

    outfile.close()


create_feature_map(features)

importance = bst.get_fscore(fmap='xgb.fmap')
importance = sorted(importance.items(), key=operator.itemgetter(1))

df = pd.DataFrame(importance, columns=['feature', 'fscore'])
df['fscore'] = df['fscore'] / df['fscore'].sum()

test_xgb = xgb.DMatrix(test_x)
submission = pd.read_csv("./input/sample_submission.csv")
submission.iloc[:, 1] = np.exp(bst.predict(test_xgb))
submission.to_csv('xgb_starter.sub.csv', index=None)
