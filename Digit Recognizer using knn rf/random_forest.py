# coding=UTF-8


import os
import sys
import re
import random
import math

from collections import Counter

from pylon import datalines
from pylon import datamatrix
from pylon import puts
from pylon import joiner
from pylon import load_csv
from pylon import statistic



from pprint import pprint


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
    # print(self.decision_tree)

  def __str__(self):
    return '<DecisionTree> {}'.format(self.decision_tree)

  def classify(self, test_feature, decision_tree=None):
    '''对新的数据进行分类'''
    if decision_tree is None: decision_tree = self.decision_tree
    if not isinstance(decision_tree, dict):
      # 决策树直接就是最后的分类结果,, 不是分叉的dict, 在训练数据全为同一类时会遇到这种情况
      return decision_tree
    root_node = list(decision_tree.keys())[0]
    # print('74', decision_tree, test_feature, root_node, self.labels, '\n')
    value_on_root_node = test_feature[self.labels.index(root_node)]
    sub_tree = decision_tree[root_node][value_on_root_node]

    if isinstance(sub_tree, dict):
      return self.classify(test_feature, sub_tree)
    else:
      return sub_tree

  def primary_class(self, klasses):
    return Counter(klasses).most_common(1)[0][0]

  def create_tree(self, data, labels):
    '''递归建立决策树'''
    labels = labels[:]
    klasses = [row[-1] for row in data]
    if len(set(klasses)) == 1:
      return klasses[0]
    if len(data[0]) == 1: # 只剩下分类这一列的情况, 这时要选频率最高的那个分类
      return self.primary_class(klasses)
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
    return max(info_gain_by_feature, key=lambda x: x[1])[0]

  def shannon_entropy(self, array):
    '''计算香农熵'''
    array = list(array)
    return - sum(v/len(array) * math.log(v/len(array), 2) for k, v in Counter(array).items())

  def filter(self, data, column, value):
    '''返回data中第n列中值为value的行, 并去除该value字段'''
    return [row[:column] + row[column+1:] for row in data if row[column] == value]














class RandomForest:
  '''随机森林'''
  def __init__(self, traindata, labels,
                     n_trees=5, n_features_per_tree=3, n_rows_power=0.5,
                     verbose=False):
    self.traindata = traindata
    self.labels = labels              # 训练数据的feature标签, 包含最后一项(分类)
    self.n_rows = len(traindata)
    self.n_features = len(labels) - 1 # 数据总计多少feature, 不含分类
    self.n_trees = n_trees

    self.n_rows_per_tree = int(self.n_rows ** n_rows_power) # 每棵树用多少行训练数据 power = 0.1~0.9
    self.n_features_per_tree = n_features_per_tree  # 每棵树随机选择多少项feature
    self.trees = []
    self.features_index_for_trees = []              # 每棵树的feature索引, 用于分类时提取测试数据中相关的字段
    self.verbose = verbose

  def train(self):
    for i in range(self.n_trees):

      features_index = random.sample(list(range(self.n_features)), self.n_features_per_tree)
      self.features_index_for_trees.append(features_index)
      data_filtered = self.filter_traindata(features_index)
      labels_filtered = self.filter_labels(features_index)
      tree = DecisionTree(data_filtered, labels=labels_filtered)
      self.trees.append(tree)
    if self.verbose:
      puts('train done: total {self.n_rows} rows, {self.n_trees} trees, each tree use {self.n_rows_per_tree} rows data')
      t1, t2, t3, *_, tn = self.trees
      pprint(t1.decision_tree)
      pprint(t2.decision_tree)
      pprint(t3.decision_tree)
      pprint('------------------------')


  def filter_traindata(self, features_index):
    '''提取数据, 按每棵树需要n项数据, 先从总训练数据中取样n行(有放回),
    再按照为该树分配的 features_index 过滤掉不需要的字段, 最后的分类字段总是保留'''
    data = random.sample(self.traindata, self.n_rows_per_tree)
    features_index_with_class = features_index + [self.n_features]
    data_filtered = []
    for row in data:
      data_filtered.append([elem for i, elem in enumerate(row) if i in features_index_with_class])
    return data_filtered

  def filter_labels(self, features_index):
    '''按照 features_index 过滤掉不需要的 label, 最后的分类总是保留'''
    features_index_with_class = features_index + [self.n_features]
    return [elem for i, elem in enumerate(self.labels) if i in features_index_with_class]


  def classify(self, testdata):
    '''为测试数据分类
    首先对应每棵树和它的 features_index
    将测试数据中属于 features_index 的字段取出, 交给该树做分类
    最后汇总所有树的分类结果'''
    result = []
    for tree, features_index in zip(self.trees, self.features_index_for_trees):
      testdata_filtered = [elem for i, elem in enumerate(testdata) if i in features_index]
      try:
        value = tree.classify(testdata_filtered)
      except KeyError as e:
        # 训练树的数据太少了, 没有涵盖字段中每一种可能出现的值, 于是构建的树不完全, 会在测试中出错
        value = 'classify failed'
      result.append(value)
    return Counter(result)























from pyshould import should


def test_check_result():
  path_result = 'result_by_self_train[28000].csv'
  path_benchmark = 'rf_benchmark.csv'
  result = load_csv(path_result, sample=0)
  # benchmark = load_csv(path_benchmark, sample=21)
  score = []
  for i, r, b in result:
    score.append('+' if str(r) == b[1:2] else '-')
  puts(statistic(score))








def write_csv(path, rows, headers=None):
  import csv
  if headers is None:
    headers = ['Value' + str(i) for i in range(len(rows[0]))]
  # rows = [('AA', 39.48, '6/11/2007', '9:36am', -0.18, 181800),
  #          ('AIG', 71.38, '6/11/2007', '9:36am', -0.15, 195500),
  #          ('AXP', 62.58, '6/11/2007', '9:36am', -0.46, 935000),
  #        ]
  with open(path, 'w') as f:
    f_csv = csv.writer(f)
    f_csv.writerow(headers)
    f_csv.writerows(rows)




def test_trim_result():
  path = 'result_by_self_train[28000].csv'
  result = []
  for i, r, b in load_csv(path, sample=0):
    result.append(int(r) if r != 'classify failed' else random.choice(list(range(0, 10))))
  puts(statistic(result))
  result = [[r] for r in result]
  write_csv('submit_' + path, rows=result, headers=['val'])





def test_check_version():
  print(sys.version)












def test_tree():
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
  data, title = datamatrix(text)
  dt = DecisionTree(data, labels=title)
  n = dt.best_split_feature(data)
  tree = dt.decision_tree
  print(n)
  puts(tree)

  test1 = dt.classify(['青年', 'no',  'no', '好'], decision_tree=tree)
  test2 = dt.classify(['青年', 'no',  'yes', '非常好'], decision_tree=tree)
  test3 = dt.classify(['老年', 'yes', 'yes', '一般'], decision_tree=tree)
  test4 = dt.classify(['老年', 'no',  'no', '好'], decision_tree=tree)
  test1    | should.eq('不给')
  test2    | should.eq('给')
  test3    | should.eq('给')
  test4    | should.eq('不给')




def generate_9gong_test(traindata, intrain=False, klass='positive'):
  if intrain:
    while True:
      test = random.choice(traindata)
      if klass == test[-1]:
        return test[:-1], 'should be '+test[-1]
      continue
  pool = list('xxxxxxxxxxoooooooooobbbbbbbbbb')
  while True:
    test = random.sample(pool, 9)
    for row in traindata:
      if row[:-1] == test: break
    else: return test, 'should be ?       '



def test_forest_9gong():
  '''九宫格测试, 9个feature, 每个feature可以是 x o b 三种值,
  数据网上找的, 大概连成一排算做 postive, 否则 negative, 我也不确定
  这个测试效果不好, 经常把 negative 误识别为 positive'''
  traindata = list(load_csv('9gong.csv', sample=0))
  labels = load_csv('9gong.csv', only_title=True)
  forest = RandomForest(traindata, labels, n_trees=100, n_features_per_tree=6)
  forest.train()

  tests_p = [generate_9gong_test(traindata, intrain=True, klass='positive')[0] for i in range(10)]
  tests_n = [generate_9gong_test(traindata, intrain=True, klass='negative')[0] for i in range(10)]

  for i, x in enumerate(tests_p):
    result = forest.classify(x)
    print(i, 'should be positive:', result)
  print('---------')
  for i, x in enumerate(tests_n):
    result = forest.classify(x)
    print(i, 'should be negative:', result)




def test_forest_9gong_using_scikit_learn():
  '''scikit的随机森林
  分的特别准, 没有一次失手'''

  def convert_to_int(seq):
    return [('x', 'o', 'b').index(elem) for elem in seq]

  from sklearn.ensemble import RandomForestClassifier
  traindata = list(load_csv('9gong.csv', sample=0))
  labels = load_csv('9gong.csv', only_title=True)
  traindataX = [row[:-1] for row in traindata]
  traindataX = [convert_to_int(row) for row in traindataX]
  traindataY = [row[-1] for row in traindata]


  print('Training...')
  rf = RandomForestClassifier(n_estimators=100)
  print('Predicting...')
  rf_model = rf.fit(traindataX, traindataY)

  tests_p = [convert_to_int(generate_9gong_test(traindata, intrain=True, klass='positive')[0]) for i in range(10)]
  tests_n = [convert_to_int(generate_9gong_test(traindata, intrain=True, klass='negative')[0]) for i in range(10)]

  for i, x in enumerate(rf_model.predict(tests_p)):
    print(i, 'should be positive:', x)
  print('---------')
  for i, x in enumerate(rf_model.predict(tests_n)):
    print(i, 'should be negative:', x)






def test_forest_vote():
  '''党派投票测试
  每个 feature 只有 yes no 两种值
  无论怎样都识别的很好'''
  traindata = list(load_csv('vote.data', sample=None))
  labels = load_csv('vote.data', only_title=True)
  forest = RandomForest(traindata, labels, n_trees=200, n_features_per_tree=3)
  forest.train()
  testdata = list(load_csv('vote.test', sample=None))
  for row in testdata:
    *test, test_should = row
    result = forest.classify(test)
    print('should be: {:<15} calculated: {}'.format(test_should, result))


def test_forest_vote_hardcore():
  '''党派投票测试, 只用训练数据, 挑一部分作为测试, 剩下的仍用于训练
  无论怎样都识别的很好'''
  data = list(load_csv('vote.data', sample=None))
  traindata, testdata = data[40:], data[:40]
  labels = load_csv('vote.data', only_title=True)
  forest = RandomForest(traindata, labels, n_trees=200, n_features_per_tree=3)
  forest.train()
  for row in testdata:
    *test, test_should = row
    result = forest.classify(test)
    print('should be: {:<15} calculated: {}'.format(test_should, result))
