
# import os
# import sys
# import fnmatch
# import re
# import inspect
# import time
# import datetime
# import itertools
# import random
# import math

# from collections import defaultdict
# from collections import namedtuple
# from collections import Counter

# # import pylon


# # from pylon import dedupe
# # from pylon import rotate
# # from pylon import list_or_tuple
# # from pylon import flatten
# from pylon import transpose
# from pylon import joiner
# # from pylon import rand
# # from pylon import all_files
# # from pylon import valid_lines
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

from collections import Counter
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from numpy import savetxt, loadtxt
train_sample = 1000
test_sample = 2000


# train_sample = 1000
# test_sample = 2000
# rf = RandomForestClassifier(n_estimators=400)
# Training...
# Predicting...
# Counter({True: 1844, False: 156}), 92.2%
# [Finished in 25.4s]







train = loadtxt('train.csv', delimiter=',',skiprows=1)
X_train = np.array([x[1:] for x in train][:train_sample])
print(X_train.shape)
Y_train = np.array([x[0] for x in train][:train_sample])
print(Y_train.shape)
X_test = loadtxt('test.csv', delimiter=',',skiprows=1)
print(X_test.shape)
print('Training...')
rf = RandomForestClassifier(n_estimators=400)
print('Predicting...')
rf_model = rf.fit(X_train,Y_train)
pred = [[index+1,x] for index,x in enumerate(rf_model.predict(X_test[:test_sample]))]


benchmark = loadtxt('rf_benchmark.csv', delimiter=',',skiprows=1, converters = {1: lambda x:int(x.decode()[1])})


correct_list = []
for i in range(test_sample):
  number = int(pred[i][1])
  bm = int(benchmark[i][1])
  correct_list.append(True if number==bm else False)
  # print('{number} == {bm}'.format(**vars()))
score = '{}, {:.1%}'.format(Counter(correct_list), Counter(correct_list)[True]/test_sample)
print(score)
# savetxt('result_by_rf_train[1000].csv',pred,delimiter=',',fmt='%d,%d',header='ImageId,Label',comments='')
# print('Done.')





