#!/usr/bin/python
# -*- coding:utf8 -*-

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import KFold
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers.advanced_activations import PReLU

np.random.seed(123)


# 神经网络模块
def nn_model():
    model = Sequential()
    model.add(Dense(400, input_dim=xtrain.shape[1], init='he_normal'))
    model.add(PReLU())
    model.add(Dropout(0.4))
    model.add(Dense(200, init='he_normal'))
    model.add(PReLU())
    model.add(Dropout(0.2))
    model.add(Dense(1, init='he_normal'))
    model.compile(loss='mae', optimizer='adadelta')
    return model


# 批量生成器
def batch_generator(X, y, batch_size, shuffle):
    # chenglong code for fiting from generator
    # numpy.ceil计算每个元素的天花板，即大于或等于每个元素的最小值
    number_of_batches = np.ceil(X.shape[0]/batch_size)
    counter = 0
    # arange函数类似于python的range函数，通过指定开始值、终值和步长来创建一维数组，数组不包括终值
    sample_index = np.arange(X.shape[0])
    if shuffle:
        np.random.shuffle(sample_index)
    while True:
        batch_index = sample_index[batch_size*counter:batch_size*(counter+1)]
        X_batch = X[batch_index, :].toarray()
        y_batch = y[batch_index]
        counter += 1
        yield X_batch, y_batch
        if (counter == number_of_batches):
            if shuffle:
                np.random.shuffle(sample_index)
            counter = 0


def batch_generatorp(X, batch_size, shuffle):
    number_of_batches = X.shape[0] / np.ceil(X.shape[0] / batch_size)
    counter = 0
    sample_index = np.arange(X.shape[0])
    while True:
        batch_index = sample_index[batch_size * counter:batch_size * (counter + 1)]
        X_batch = X[batch_index, :].toarray()
        counter += 1
        yield X_batch
        if (counter == number_of_batches):
            counter = 0

# 主程序
# read data  读取数据
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
train = train.head(50000)
# set test loss to NaN 在test里创建loss这一列，并设值为Nan
test['loss'] = np.nan

# response and IDs 提取train里的id和loss，以及test的id
y = train['loss'].values
id_train = train['id'].values
id_test = test['id'].values

# stack train test 设ntrain为train的行数，并拼接train和test
ntrain = train.shape[0]
tr_te = pd.concat((train, test), axis=0)

# 建立稀疏矩阵
sparse_data = []

f_cat = [f for f in tr_te.columns if 'cat' in f]
for f in f_cat:
    dummy = pd.get_dummies(tr_te[f].astype('category'))
    tmp = csr_matrix(dummy)
    sparse_data.append(tmp)

f_num = [f for f in tr_te.columns if 'cont' in f]
# 使用该类的好处在于可以保存训练集中的参数（均值、方差）直接使用其对象转换测试集数据
scaler = StandardScaler()
tmp = csr_matrix(scaler.fit_transform(tr_te[f_num]))
sparse_data.append(tmp)
# sparse train and test data
# hstack将稀疏矩阵横向或者纵向合并
xtr_te = hstack(sparse_data, format='csr')
xtrain = xtr_te[:ntrain, :]
xtest = xtr_te[ntrain:, :]

print('Dim train', xtrain.shape)
print('Dim test', xtest.shape)

# cv-folds 将样例划分为K份，若K=len(样例)，即为留一交叉验证，K-1份作为训练。从sklearn中自带的KFold函数说明中也可以看到其用法。其中n_folds默认为3折交叉验证，2/3作为训练集，1/3作为测试集。
nfolds = 5
folds = KFold(len(y), n_folds=nfolds, shuffle=True, random_state=111)

# train models
i = 0
nbags = 5
nepochs = 55
pred_oob = np.zeros(xtrain.shape[0])
pred_test = np.zeros(xtest.shape[0])

for (inTr, inTe) in folds:
    xtr = xtrain[inTr]
    ytr = y[inTr]
    xte = xtrain[inTe]
    yte = y[inTe]
    pred = np.zeros(xte.shape[0])
    for j in range(nbags):
        model = nn_model()
        # 利用Python的生成器，逐个生成数据的batch并进行训练。生成器与模型将并行执行以提高效率。例如，该函数允许我们在CPU上进行实时的数据提升，同时在GPU上进行模型训练
        fit = model.fit_generator(generator=batch_generator(xtr, ytr, 128, True),
                                  nb_epoch=nepochs,
                                  samples_per_epoch=xtr.shape[0],
                                  verbose=0)
        # 从一个生成器上获取数据并进行预测，生成器应返回与predict_on_batch输入类似的数据
        pred += model.predict_generator(generator=batch_generatorp(xte, 800, False),
                                        val_samples=xte.shape[0])[:, 0]
        pred_test += model.predict_generator(generator=batch_generatorp(xtest, 800, False),
                                             val_samples=xtest.shape[0])[:, 0]
    pred /= nbags
    pred_oob[inTe] = pred
    score = mean_absolute_error(yte, pred)
    i += 1
    print('Fold ', i, '- MAE:', score)

print('Total - MAE:', mean_absolute_error(y, pred_oob))

# train predictions
df = pd.DataFrame({'id': id_train, 'loss': pred_oob})
df.to_csv('preds_oob.csv', index=False)

# test predictions
pred_test /= (nfolds*nbags)
df = pd.DataFrame({'id': id_test, 'loss': pred_test})
df.to_csv('result.csv', index=False)
