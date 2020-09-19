from surprise import Dataset
from surprise import Reader
from surprise import accuracy
from surprise.model_selection import split
from surprise import SVD,SVDpp
import time

# 数据读取
reader = Reader(line_format='user item rating timestamp', sep=',', skip_lines=1)
data = Dataset.load_from_file(r'C:\Users\yy\Desktop\BI\L4\L4-2\L4-code\MovieLens\ratings.csv', reader=reader)
train_s,test_s = split.train_test_split(data, train_size=0.8)

algo1 = SVD(biased = False)
algo2 = SVD()
algo3 = SVDpp()

"""SVD"""
print('SVD结果：')
time1=time.time()
algo1.fit(train_s)
pred = algo1.test(test_s)
accuracy.rmse(pred, verbose=True)
time2=time.time()
print('SVD用时: %.2fs' % (time2-time1))
uid = str(196)
iid = str(302)
algo1.predict(uid, iid, r_ui=4, verbose=True)   # 输出uid对iid的预测结果
print('-'*30)

"""SVDbias"""
print('SVDbias结果:')
time1=time.time()
algo2.fit(train_s)
pred = algo2.test(test_s)
accuracy.rmse(pred, verbose=True)
time2=time.time()
print('SVDbias用时: %.2fs' % (time2-time1))
uid = str(196)
iid = str(302)
algo2.predict(uid, iid, r_ui=4, verbose=True)
print('-'*30)

"""SVD++"""
print('SVD++结果:')
time1=time.time()
algo3.fit(train_s)
pred = algo3.test(test_s)
accuracy.rmse(pred, verbose=True)
time2=time.time()
print('SVD++用时: %.2fs' % (time2-time1))
uid = str(196)
iid = str(302)
algo3.predict(uid, iid, r_ui=4, verbose=True)
print('-'*30)
