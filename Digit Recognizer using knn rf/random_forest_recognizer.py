
import time

from collections import Counter



from pylon import load_csv
from pylon import puts




def cut(array, max=1):
  return [max if x > max else x for x in array]



# trainset = [cut(row[1:], max=1) + [row[0]] for row in load_csv('train.csv', sample=1000)]
# testset = [cut(row, max=1) for row in load_csv('test.csv', sample=2000)]
# benchmarkset = list(load_csv('rf_benchmark.csv', sample=2000))
# forest = RandomForest(trainset, labels, n_trees=400, n_features_per_tree=70)
# classify done Counter({'+': 1801, '-': 199}), 90.0%

def test_run_forest():
  trainset_sample = None and 1000
  testset_sample = None and 30

  trainset = [cut(row[1:], max=1) + [row[0]] for row in load_csv('train.csv', sample=trainset_sample)]
  labels = ['pixel'+str(i) for i in range(784)] + ['number']
  testset = [cut(row, max=1) for row in load_csv('test.csv', sample=testset_sample)]
  benchmarkset = list(load_csv('rf_benchmark.csv', sample=testset_sample))



  from random_forest import RandomForest


  def run_forest(n_trees=400, n_features_per_tree=70, n_rows_power=0.5):
    start_time = time.time()
    forest = RandomForest(trainset, labels, n_trees, n_features_per_tree, n_rows_power)
    forest.train()
    score = []
    for test, (i, bm) in zip(testset, benchmarkset):
      result = forest.classify(test)
      result = result.most_common(1)[0][0]
      print('{:.1f}s -- No.{} should be: <{}> cal: {}'.format(time.time()-start_time, i, bm, result))
      if result == bm:
        score.append('+')
      else:
        score.append('-')
      with open('result_by_self_train[28000].csv', 'a') as f:
        f.write('{},{},<{}>\n'.format(i, result, bm))
    # print('classify done {}, {:.1%}'.format(Counter(score), Counter(score)['+'] / len(score)))

    return Counter(score)['+'] / len(score)


  # run_forest(n_trees=300, n_features_per_tree=30, n_rows_power=0.5)







def brute_force_params(run_func, logger, **params_range):
  '''尝试参数的各种可能组合
  每个输入参数是一个可能值的列表, 比如 range(0, 250, 10)
  使用 itertools.product 遍历所有可能的参数组'''
  from itertools import product
  import time
  total_start = time.time()
  total = []
  params_product = list(product(*[[(k, elem) for elem in v] for k, v in params_range.items()]))
  puts('params prepared, {} variantions'.format(len(params_product)))
  for i, params in enumerate(params_product):
    loop_start = time.time()

    result = run_func(**dict(params))
    # result = 0
    total.append([params, result])
    loop_cost = time.time() - loop_start
    total_cost = time.time() - total_start
    percent = (i+1) / len(params_product)
    info = 'get <{result:.3%}> by: {params} \n    [current {loop_cost:.1f}s / total {total_cost:.1f}s] {percent:.2%}'.format_map(vars())
    logger.debug(info)

  puts('all done!')
  logger.debug(total)
  return total







# logger = create_logger('random_forest_recognizer_brute_force_params')
# brute_force_params(run_forest, logger, n_trees=range(250, 500, 60),
#                                # n_features_per_tree=range(40, 200, 10),
#                                n_features_per_tree=[30],
#                                n_rows_power=[round(x*0.1, 2) for x in range(7, 10, 1)])

# n_features_per_tree = 30 差不多就够, 多了没用
# n_trees = 200~300 差不多就够
# n_rows_power 貌似越多越好


def test_random_forest_recognizer_brute_force_params_parse1():
  path = 'random_forest_recognizer_brute_force_params.log'
  print(path)



def test_random_forest_recognizer_brute_force_params_parse():
  '''分析不同参数组合的成绩'''
  from pylon import datalines_from_file
  path = 'random_forest_recognizer_brute_force_params.log'
  print(path)
  for row in datalines_from_file(path):
    if row.startswith('--') and row.startswith('['):
      continue
    if not "'n_rows_power', 0.7" in row:
      continue
    print(row)

