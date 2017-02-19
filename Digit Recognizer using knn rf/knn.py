
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





def distance(self, other):
  return sum((a-b)*(a-b) for a,b in zip(self, other))**0.5
def cut(array, max=1):
  return [max if x > max else x for x in array]



def classify(d, train_data, labels, k=3):
  data_labels = zip(train_data, labels)
  most_nears = sorted(data_labels, key=lambda x: distance(x[0], d))[0:k]
  # print(most_nears)
  result = Counter(x[1] for x in most_nears)
  # print(result)
  return result.most_common(1)[0][0]




if __name__ == '__main__':
  trainset = [(row[0], cut(row[1:], max=1)) for row in load_csv('train.csv', sample=100)]
  labels, train_data = transpose(trainset)

  testset = list(load_csv('test.csv', sample=50))
  benchmarkset = list(load_csv('rf_benchmark.csv', sample=50))


  def find_k(mink=3, maxk=15):
    for k in range(mink, maxk+1):
      result = []
      for test, (index, bm) in zip(testset, benchmarkset):
        number = classify(test, train_data, labels, k=k)
        if index < 5:
          print('index {}: get {} (bm {})'.format(index, number, bm))
        if number == bm: result.append('correct')
        else: result.append('miss')

      counter = Counter(result)
      print('{}, total {}, error {:.2%}, k={}'.format(counter, len(result), counter['miss']/len(result), k))


  # find_k(mink=3, maxk=15)
  # find_k(mink=9, maxk=9)



