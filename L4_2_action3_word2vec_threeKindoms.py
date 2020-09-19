# -*- coding: utf-8 -*-
"""
Created on Sat Sep 19 17:36:59 2020

@author: yy
"""

"""1. 对小说进行文本分割"""
import jieba
import os
from utils import files_processing

# 源文件所在目录
source_folder = r'C:\Users\yy\Desktop\BI\L4\L4-2\L4-code\word2vec\three_kingdoms\source'

# 分词结果存放目录
segment_folder = r'C:\Users\yy\Desktop\BI\L4\L4-2\L4-code\word2vec\three_kingdoms\segment'

# 对整个文件内容进行字词分割
def segment_lines(file_list, segment_out_dir, stopwords=[]):
    for i,file in enumerate(file_list):
        segment_out_name=os.path.join(segment_out_dir,'segment_{}.txt'.format(i))
        with open(file, 'rb') as f:
            document = f.read()
            document_cut = jieba.cut(document)
            sentence_segment=[]
            for word in document_cut:
                if word not in stopwords:
                    sentence_segment.append(word)
            result = ' '.join(sentence_segment)
            result = result.encode('utf-8')
            with open(segment_out_name, 'wb') as f2:
                f2.write(result)

# 对source中的txt文件进行分词，输出到segment目录中
file_list = files_processing.get_files_list(source_folder, postfix='*.txt')
segment_lines(file_list, segment_folder)


"""2. 将小说的分割结果转化成vec，计算单词相似度"""
from gensim.models import word2vec
import multiprocessing

# 如果目录中有多个文件，可以使用PathLineSentences
sentences = word2vec.PathLineSentences(segment_folder)

# 设置模型参数，进行训练
model = word2vec.Word2Vec(sentences, size=100, window=3, min_count=1)

# 读取三国人物列表
file = r'C:\Users\yy\Desktop\BI\L4\L4-2\L4-code\word2vec\three_kingdoms\characters.txt'
chara_list = []
with open(file, 'r') as f:
    for line in f.readlines():
        line = line.strip()
        chara_list.extend(line.split('、'))

# 计算曹操和每个人物的相似度
charac_similarity = []
for i in range(len(chara_list)):
    try:
        sim = model.wv.similarity('曹操', chara_list[i])
        charac_similarity.append((chara_list[i], sim))
    except (KeyError):
        continue

# 对人物按相似度从大到小排序
charac_similarity = sorted(charac_similarity, key = lambda x:x[1], reverse = True)

# 找到top-k个和曹操相似的人物
k = 5
print('和曹操最相近的top-%d人物有：' % k)
for i in range(k):
    print(charac_similarity[i])         
            
print('-'*30)           
print('曹操+刘备-张飞=')
print(model.wv.most_similar(positive=['曹操', '刘备'], negative=['张飞']))


