
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

from collections import defaultdict
from collections import namedtuple
from collections import Counter

# import pylon


# from pylon import dedupe
# from pylon import rotate
# from pylon import list_or_tuple
# from pylon import flatten
from pylon import transpose
from pylon import joiner
# from pylon import rand
# from pylon import all_files
# from pylon import valid_lines
# from pylon import valid_lines_from_file
# from pylon import file_timer
# from pylon import sleep
# from pylon import warn
# from pylon import microtest
# from pylon import create_logger
# from pylon import puts
# from pylon import watch
# from pylon import trace
# from pylon import nested_property

import numpy as np

a = np.array([1,2,3])
b = np.array([1,2,2])
c = np.array([[1,3], [1,2]])

# print(c)
# print(c[0])

import csv

def load_csv(path, sample=10, title=True):
  with open(path) as file:
    lines = csv.reader(file)
    if title:
      next(lines)
    for i, line in enumerate(lines, 1):
      if sample and i > sample:
        return
      yield [int(x) for x in line]









# def cut(array, max=1):
#   return array
#   return [max if x > max else x for x in array]



def classify(d, train_data, labels, k=3):
  size = train_data.shape[0]
  distances = ((d - train_data)**2).sum(axis=1)**0.5
  distances_sort = distances.argsort()
  most_nears = [labels[distances_sort[i]] for i in range(k)]
  result = Counter(most_nears)
  return result.most_common(1)[0][0]

def report(result):
  counter = Counter(result)
  print('{}, total {}, error {:.2%}'.format(counter, len(result), counter['miss']/len(result)))




if __name__ == '__main__':
  trainset = [(row[0], row[1:]) for row in load_csv('train.csv', sample=6000)]
  labels, train_data = transpose(trainset)
  labels = np.array(labels)
  train_data = np.array(train_data)


  testset = np.array(list(load_csv('test.csv', sample=None)))
  benchmarkset = np.array(list(load_csv('rf_benchmark.csv', sample=None)))


  def find_k(mink=3, maxk=15):
    file = open('result.csv', 'w')
    file.write('index, result, benchmark, correct\n')
    for k in range(mink, maxk+1):
      result = []
      # print(testset)
      # print(benchmarkset)
      test_size = testset.shape[0]

      for index in range(test_size):
        bm = benchmarkset[index][1]
        number = classify(testset[index], train_data, labels, k=k)
        if index % 100 == 1:
          print('index {}: get {} (bm {})'.format(index, number, bm))
          report(result)
        if number == bm: description = 'correct'
        else: description = 'miss'
        result.append(description)
        file.write('{index}, {number}, {bm}, {description}\n'.format(**vars()))

      report(result)

  # find_k(mink=3, maxk=15)
  find_k(mink=5, maxk=5)



