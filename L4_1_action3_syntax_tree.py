# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 17:57:36 2020

@author: yy
"""

import random

# 定义语法从句
grammar = """
host = 问候 ， 自我介绍 询问 具体业务 结尾
问候 = 称谓 打招呼
称谓 = 先生 | 女士
打招呼 = 你好 | 您好 
自我介绍 = 我是开课吧 姓氏 老师 ，
姓氏 = 赵 | 钱 | 孙 | 李 | 周 | 吴 | 郑 | 王
询问 = 请问 对象 咨询
对象 = 你要 | 您需要
具体业务 = 数据分析 | 前端开发 | 推荐系统 | NLP | CV
结尾 = 的课程吗？"""


# 得到语法字典
def getGrammarDict(gram, linesplit = "\n", gramsplit = "="):
    #定义字典
    result = {}

    for line in gram.split(linesplit):
        # 去掉首尾空格后，如果为空则退出
        if not line.strip(): 
            continue
        expr, statement = line.split(gramsplit)
        result[expr.strip()] = [i.split() for i in statement.split("|")]
    #print(result)
    return result

# 生成句子
def generate(gramdict, target, isEng = False):
    if target not in gramdict: 
        return target
    find = random.choice(gramdict[target])
    #print(find)
    blank = ''
    # 如果是英文中间间隔为空格
    if isEng: 
        blank = ' '
    return blank.join(generate(gramdict, t, isEng) for t in find)

gramdict = getGrammarDict(grammar)
print(generate(gramdict, "host"))
