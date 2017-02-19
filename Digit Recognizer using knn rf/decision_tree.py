
import os
import sys
import fnmatch
import re
import inspect
import time
import datetime
import itertools
import random
import math

# from collections import defaultdict
# from collections import namedtuple
from collections import Counter

# # import pylon

# # from pylon import dedupe
# # from pylon import rotate
# # from pylon import list_or_tuple
# # from pylon import flatten
# from pylon import transpose
# from pylon import joiner
# # from pylon import rand
from pylon import datalines
# from pylon import valid_lines
# # from pylon import valid_lines_from_file
# # from pylon import file_timer
# # from pylon import sleep
# # from pylon import warn
# # from pylon import microtest
# # from pylon import create_logger
# # from pylon import puts
# # from pylon import watch
# # from pylon import trace
# # from pylon import nested_property

# Decision tree




def calcShannonEnt(dataSet):
  '''计算熵
  找出有几种目标指数
  根据他们出现的频率计算其信息熵'''
  numEntries=len(dataSet)
  labelCounts={}
  for featVec in dataSet:
    currentLabel=featVec[-1]
    if currentLabel not in labelCounts.keys():
      labelCounts[currentLabel]=0
    labelCounts[currentLabel]+=1

  shannonEnt=0.0
  for key in labelCounts:
     prob = float(labelCounts[key]) / numEntries
     shannonEnt -= prob * math.log(prob, 2)

  return shannonEnt



def splitDataSet(dataSet, axis, value):
  '''分割数据
  因为要每个特征值都计算相应的信息熵，所以要对数据集分割，将所计算的特征值单独拿出来'''
  retDataSet = []
  for featVec in dataSet:
    if featVec[axis] == value:
      reducedFeatVec = featVec[:axis] # chop out axis used for splitting
      reducedFeatVec.extend(featVec[axis+1:])
      retDataSet.append(reducedFeatVec)
  return retDataSet



def chooseBestFeatureToSplit(dataSet):
  '''找出信息熵增益最大的特征值'''
  numFeatures = len(dataSet[0]) - 1    # the last column is used for the labels
  baseEntropy = calcShannonEnt(dataSet)
  bestInfoGain = 0.0; bestFeature = -1
  for i in range(numFeatures):     # iterate over all the features
    featList = [example[i] for example in dataSet] # create a list of all the examples of this feature
    uniqueVals = set(featList)     # get a set of unique values

    newEntropy = 0.0
    for value in uniqueVals:
      subDataSet = splitDataSet(dataSet, i, value)
      prob = len(subDataSet)/float(len(dataSet))
      newEntropy += prob * calcShannonEnt(subDataSet)
    infoGain = baseEntropy - newEntropy   # calculate the info gain; ie reduction in entropy

    if (infoGain > bestInfoGain):     # compare this to the best gain so far
      bestInfoGain = infoGain     # if better than current best, set to best
      bestFeature = i
  return bestFeature            # returns an integer




# 喉结(1 有喉结) 胡子(1 有胡子) 性别(1 男)
# 1 0 1
# 1 1 1
# 0 1 1
# 0 0 0

# 结果是输出0，也就是是否有喉结对性别影响最大。
# data = [
#   [1, 0, 'man'],
#   [1, 1, 'man'],
#   [0, 1, 'man'],
#   [0, 0, 'woman'],
# ]
# n = chooseBestFeatureToSplit(data)
# print(n)








# 接着上一节说，没看到请先看一下上一节关于数据集的划分数据集划分。现在我们得到了每个特征值的信息熵增益，我们按照信息熵增益的从大到校的顺序，安排排列为二叉树的节点。数据集和二叉树的图见下。

#     throat
#       / \
#      0   1
#     /     \
#   mustache  man
#     / \
#    0   1
#   /     \
# woman   man

# 上一节我们通过chooseBestFeatureToSplit函数已经可以确定当前数据集中的信息熵最大的那个特征值。我们将最大的那个作为决策树的父节点，这样递归下去就可以了。

def createTree(dataSet, labels):
  # 把所有目标指数放在这个list里
  classList = [example[-1] for example in dataSet]
  # 下面两个if是递归停止条件，分别是list中都是相同的指标或者指标就剩一个。
  if classList.count(classList[0]) == len(classList):
    return classList[0]
  if len(dataSet[0]) == 1:
    return majorityCnt(classList)
  # 获得信息熵增益最大的特征值
  bestFeat = chooseBestFeatureToSplit(dataSet)
  bestFeatLabel = labels[bestFeat]
  # 将决策树存在字典中
  myTree = {bestFeatLabel:{}}
  # labels删除当前使用完的特征值的label
  del(labels[bestFeat])
  featValues = [example[bestFeat] for example in dataSet]
  uniqueVals = set(featValues)
  # 递归输出决策树
  for value in uniqueVals:
    subLabels = labels[:]     # copy all of labels, so trees don't mess up existing labels

    myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
  return myTree

# tree = createTree(data, ['喉结', '胡子'])
# print(tree)


# 打印出来的决策树：
# {'throat': {0: {'mustache': {0: 'women', 1: 'man'}}, 1: 'man'}}


# 下面就是如何是用建立好的决策树。我们建立函数
def classify(inputTree, featLabels, testVec):
  '''
  inputTree：是输入的决策树对象
  featLabels：是我们要预测的特征值得label，如：['throat','mustache']
  testVec:是要预测的特征值向量，如[0,0]
  '''
  # 存储决策树第一个节点
  firstStr = list(inputTree.keys())[0]
  # 将第一个节点的值存到secondDict字典中
  secondDict = inputTree[firstStr]
  # 建立索引，知道对应到第几种特征值
  featIndex = featLabels.index(firstStr)
  key = testVec[featIndex]
  valueOfFeat = secondDict[key]
  # 对比，判断当前的键值是否是一个dict类型，如果是就递归，不是就输出当前键值为结果
  if isinstance(valueOfFeat, dict):
    classLabel = classify(valueOfFeat, featLabels, testVec)
  else: classLabel = valueOfFeat
  return classLabel


# print(classify(tree, ['喉结', '胡子'], [0, 0]))
# 测试：当我们输入classify(mtree,['throat','mustache'],[0,0])时，显示结果是women，表明没有喉结和胡子是女人。












class DecisionTree:
  '''基本的决策树模型
  可以处理离散型的数据,
  没有实现剪枝等功能
  输入一定量训练数据后, 对一个新的测试数据进行分类'''
  def __init__(self, traindata, labels):
    '''
    labels 是所有的指标以及分类, 其中最后一个 label 是分类
    traindata 是二维表, 每行代表一个已知的数据,
    开始的n-1个列是在各项指标下的值, 最后一列是分类'''
    self.data = traindata
    self.labels = labels
    self.decision_tree = self.create_tree(self.data, self.labels)

  def classify(self, test_feature, decision_tree):
    '''对新的数据进行分类'''
    root_node = list(decision_tree.keys())[0]
    value_on_root_node = test_feature[self.labels.index(root_node)]
    sub_tree = decision_tree[root_node][value_on_root_node]

    if isinstance(sub_tree, dict):
      return self.classify(test_feature, sub_tree)
    else:
      return sub_tree

  def create_tree(self, data, labels):
    '''递归建立决策树'''
    labels = labels[:]
    klasses = [row[-1] for row in data]
    if len(set(klasses)) == 1:
      return klasses[0]
    if len(data[0]) == 1:
      raise
    best_column = self.best_split_feature(data)
    best_label = labels[best_column]

    tree = {best_label: {}}
    del(labels[best_column])

    for unique_value in self.unique_values(data, best_column):
      sub_labels = labels[:]
      sub_data = self.filter(data, column=best_column, value=unique_value)
      tree[best_label][unique_value] = self.create_tree(sub_data, sub_labels)
    return tree

  def unique_values(self, data, column):
    '''返回data中第n列的所有可能的值'''
    return set(row[column] for row in data)

  def best_split_feature(self, data):
    '''根据最大的信息增益, 返回data中应以哪一列进行分裂'''
    base_entropy = self.shannon_entropy(row[-1] for row in data)
    feature_size = len(data[0]) - 1
    info_gain_by_feature = []

    for i in range(feature_size):
      new_entropy = 0
      for unique_value in self.unique_values(data, i):
        sub_data = self.filter(data, column=i, value=unique_value)
        new_entropy += len(sub_data) / len(data) * self.shannon_entropy(row[-1] for row in sub_data)
      info_gain_by_feature.append((i, base_entropy - new_entropy))
    # print(info_gain_by_feature)
    return max(info_gain_by_feature, key=lambda x: x[1])[0]

  def shannon_entropy(self, array):
    '''计算香农熵'''
    array = list(array)
    return - sum(v/len(array) * math.log(v/len(array), 2) for k, v in Counter(array).items())

  def filter(self, data, column, value):
    '''返回data中第n列中值为value的行, 并去除该value字段'''
    return [row[:column] + row[column+1:] for row in data if row[column] == value]







if __name__ == '__main__':

  import unittest
  from pyshould import should


  def datamatrix(text, title=True):
    ''' 多行文本转换为二维数据, 自动识别数字
    # d = "
    # Col1,Col2,Col3,中文列
    # 1,2,1,female
    # 2,3,1,male
    # 3,2,6.03,male
    # "
    # mat, title = datamatrix(d, title=True)
    # mat   | should.eq([[1,2,1,'female'],[2,3,1,'male'],[3,2,6.03,'male'],])
    # title | should.eq(['Col1','Col2','Col3','中文列'])
    '''
    def convert(text):
      if text.isdigit(): return int(text)
      try: return float(text)
      except ValueError as e: return text

    data = iter(datalines(text))
    if title: title = re.split(r'\s*,\s*', next(data))
    else: title = []

    array = []
    for line in data:
      line = [convert(e) for e in re.split(r'\s*,\s*', line)]
      array.append(line)
    return array, title





  class TestRF(unittest.TestCase):
    def setUp(self):
      self.data = [
        [1, 0, 'male'],
        [1, 1, 'male'],
        [0, 1, 'male'],
        [0, 0, 'female'],
      ]


      self.mat = '''
      Col1,Col2,Col3,中文列
      1,2, \t 1,female
      2,3,  1,male
      3,2,6.03,male
      '''

    def Dtest_read_mat(self):
      mat, title = datamatrix(self.mat, title=True)
      mat   | should.eq([[1,2,1,'female'],[2,3,1,'male'],[3,2,6.03,'male'],])
      title | should.eq(['Col1','Col2','Col3','中文列'])
      # print(mat)

    def Dtest_rf(self):

      chooseBestFeatureToSplit(self.data)     | should.be(0)
      tree = createTree(self.data, ['喉结', '胡子'])
      tree | should.eq({'喉结': {0: {'胡子': {0: 'female', 1: 'male'}}, 1: 'male'}})
      classify(tree, ['喉结', '胡子'], [0, 0]) | should.be('female')

    def test_rf2(self):
      text = '''
      年龄, 工作, 已婚,   信用, 可以借贷?
      青年, no ,  no ,   一般   , 不给
      青年, no ,  no ,   好    , 不给
      青年, yes,  no ,   好    , 给
      青年, yes,  yes,   一般   , 给
      青年, no ,  no ,   一般   , 不给
      中年, no ,  no ,   一般   , 不给
      中年, no ,  no ,   好    , 不给
      中年, yes,  yes,   好    , 给
      中年, no ,  yes,   非常好 , 给
      中年, no ,  yes,   非常好 , 给
      老年, no ,  yes,   非常好 , 给
      老年, no ,  yes,   好    ,  给
      老年, yes,  no ,   好    ,  给
      老年, yes,  no ,   非常好 , 给
      老年, no ,  no ,   一般   , 不给
      '''








      from pprint import pprint

      data, title = datamatrix(text)
      n = chooseBestFeatureToSplit(data)
      tree = createTree(data, title)

      n        | should.be(2)
      tree     | should.eq({'已婚': {'no': {'工作': {'no': '不给', 'yes': '给'}}, 'yes': '给'}})

      test1 = classify(tree, ['年龄', '工作', '已婚', '信用'], ['青年', 'no', 'no', '好'])
      test2 = classify(tree, ['年龄', '工作', '已婚', '信用'], ['青年', 'no', 'yes', '非常好'])
      test3 = classify(tree, ['年龄', '工作', '已婚', '信用'], ['老年', 'yes','yes', '一般'])
      test4 = classify(tree, ['年龄', '工作', '已婚', '信用'], ['老年', 'no', 'no', '好'])
      test1    | should.eq('不给')
      test2    | should.eq('给')
      test3    | should.eq('给')
      test4    | should.eq('不给')


      data, title = datamatrix(text)
      dt = DecisionTree(data, labels=title)
      n = dt.best_split_feature(data)
      tree = dt.decision_tree
      print(n)
      pprint(tree)

      test1 = dt.classify(['青年', 'no', 'no', '好'], decision_tree=tree)
      test2 = dt.classify(['青年', 'no', 'yes', '非常好'], decision_tree=tree)
      test3 = dt.classify(['老年', 'yes','yes', '一般'], decision_tree=tree)
      test4 = dt.classify(['老年', 'no', 'no', '好'], decision_tree=tree)
      test1    | should.eq('不给')
      test2    | should.eq('给')
      test3    | should.eq('给')
      test4    | should.eq('不给')








  unittest.main()

