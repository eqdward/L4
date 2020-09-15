# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 15:21:15 2020

@author: yy
"""

import pandas as pd
from surprise import Reader
from surprise import Dataset

from surprise import BaselineOnly, SlopeOne
from surprise import accuracy
from surprise.model_selection import KFold

# 读取items，本例为电影信息
def read_items(file_path):
    data = pd.read_csv(file_path, encoding = 'gb18030')
    id_to_name = {}
    name_to_id = {}
    for i in range(len(data['movieId'])):
        id_to_name[data['movieId'][i]] = data['title'][i]
        name_to_id[data['title'][i]] = data['movieId'][i]

    return id_to_name, name_to_id

file_path = (r'C:\Users\yy\Desktop\BI\L4\L4-1\L4-code\MovieLens\movies.csv') 
id_to_name, name_to_id = read_items(file_path)

# 数据读取
reader = Reader(line_format='user item rating timestamp', sep=',', skip_lines=1)
data = Dataset.load_from_file(r'C:\Users\yy\Desktop\BI\L4\L4-1\L4-code\MovieLens\ratings.csv', reader=reader)
# train_set = data.build_full_trainset()


"""方法1：使用SlopeOne推荐算法"""
# 定义SlopeOne算法
algo = SlopeOne()

# 定义K折交叉验证迭代器，K=5
kf = KFold(n_splits=5)
for trainset, testset in kf.split(data):   
    # 训练并预测
    algo.fit(trainset)
    predictions = algo.test(testset)
    # 计算RMSE
    accuracy.rmse(predictions, verbose=True)   #RMSE: 0.8653

# 对指定用户和商品进行评分预测
uid = str(196) 
iid = str(302) 
pred = algo.predict(uid, iid, r_ui=4, verbose=True)


"""方法2：使用baseline算法"""
# Baseline算法，使用ALS进行优化，迭代次数5，reg_u为user正则化系数为12，reg_i为item正则化系数为5
bsl_options = {'method': 'als','n_epochs': 5,'reg_u': 12,'reg_i': 5}
algo2 = BaselineOnly(bsl_options = bsl_options)

kf = KFold(n_splits=5)
for trainset, testset in kf.split(data):
    # 训练并预测
    algo2.fit(trainset)
    predictions = algo2.test(testset)
    # 计算RMSE
    accuracy.rmse(predictions, verbose=True)
    
uid = str(196)
iid = str(302)
# 输出uid对iid的预测结果测结果
pred = algo2.predict(uid, iid, r_ui=4, verbose=True)
