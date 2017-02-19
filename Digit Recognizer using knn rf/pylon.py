###################################################
############### Version 2015.05.21 ################
############################ Probe ################



import os
import sys
import re
import time
import random
import traceback



# from pprint import pprint
# import random
# import math
# import fnmatch
# import inspect
# import datetime
# import itertools

# from collections import defaultdict
# from collections import namedtuple
# from collections import Counter
# from collections import deque

# from pprint import pprint

# from pylon import dedupe
# from pylon import rotate
# from pylon import list_or_tuple
# from pylon import flatten
# from pylon import transpose
# from pylon import rand
# from pylon import listrange
# from pylon import all_files
# from pylon import datalines
# from pylon import datamatrix
# from pylon import puts
# from pylon import file_timer
# from pylon import sleep
# from pylon import warn
# from pylon import microtest
# from pylon import create_logger
# from pylon import nested_property
# from pylon import windows
# from pylon import transpose
# from pylon import random_split
# from pylon import load_title
# from pylon import load_csv




'''

if __name__ == '__main__':
  import unittest
  from pyshould import *
  import pyshould.patch

  # result | should.be_integer()
  # (1+1) | should_not.equal(1)
  # "foo" | should.equal('foo')
  # len([1,2,3]) | should.be_greater_than(2);
  # result | should.equal(1/2 + 5)
  # 1 | should_not.eq(2)
  # # Matchers not requiring a param can skip the call parens
  # True | should.be_truthy
  # # Check for exceptions with the context manager interface
  # with should.throw(TypeError):
  #   raise TypeError('foo')
  # with should.not_raise:  # will report a failure
  #   fp = open('does-not-exists.txt')
  # # Apply our custom logic for a test
  # 'FooBarBaz' | should.pass_callback(lambda x: x[3:6] == 'Bar')
  # should.be_an_integer.or_string.and_equal(1)
  # # (integer) OR (string AND equal 1)
  # should.be_an_integer.or_a_float.or_a_string
  # # (integer) OR (float) OR (string)
  # should.be_an_integer.or_a_string.and_equal_to(10).or_a_float
  # # (integer) OR (string AND equal 10) OR (float)
  # should.be_an_integer.or_a_string.but_less_than(10)
  # # (integer OR string) AND (less than 10)
  # # Note: we can use spacing to make them easier to read
  # should.be_an_integer  .or_a_string.and_equal(0)  .or_a_float
  # # (integer) OR (string AND equal 0) OR (float)
  # # Note: in this case we use capitalization to make them more obvious
  # should.be_an_integer .Or_a_string.And_equal(1) .But_Not_be_a_float
  # # ( (integer) OR (string AND equal 1) ) AND (not float)
  # # Note: if no matchers are given the last one is used
  # should.be_equal_to(10).Or(20).Or(30)
  # # (equal 10) OR (equal 20) OR (equal 30)
  # # Note: If no combinator is given AND is used by default
  # should.integer.greater_than(10).less_than(20)
  # # (integer) AND (greater than 10) AND (less than 20)
  # # Note: But by using should_either we can set OR as default
  # should_either.equal(10).equal(20).equal(30)

  class TestSequenceFunctions1(unittest.TestCase):
    def setUp(self):
      self.seq = list(range(10))
    def test_shuffle(self):
      # make sure the shuffled sequence does not lose any elements
      random.shuffle(self.seq)
      self.seq.sort()
      self.assertEqual(self.seq, list(range(10)))
      # should raise an exception for an immutable sequence
      self.assertRaises(TypeError, random.shuffle, (1,2,3))
    def test_choice(self):
      element = random.choice(self.seq)
      a = 10
      a | should.gt(20)
    def test_sample(self):
      with self.assertRaises(ValueError):
        random.sample(self.seq, 20)
      for element in random.sample(self.seq, 5):
        self.assertTrue(element in self.seq)

  unittest.main()

'''







































"""--------------
处理列表和字典
--------------"""


def dedupe(items, key=None):
  '''去除重复，可以保留原list的顺序'''
  seen = set()
  for item in items:
    val = item if key is None else key(item)
    if val not in seen:
      yield item
      seen.add(val)


def rotate(array, n):
  '''list 滑动元素的顺序，n可以是负数'''
  n = n % len(array)
  return array[n:] + array[:n]


def _list_or_tuple(x):
  return isinstance(x, (list, tuple))


def flatten(sequence, to_expand=_list_or_tuple):
  ''' 展平嵌套的list '''
  iterators = [iter(sequence)]
  while iterators:
    # 循环当前的最深嵌套（最后）的迭代器
    for item in iterators[-1]:
      # print(item)
      if to_expand(item):
        # 找到了子序列，循环子序列的迭代器
        iterators.append(iter(item))
        break
      else:
        yield item
    else:
      # 最深嵌套的迭代器耗尽，回过头来循环它的父迭代器
      iterators.pop()


def lrange(n, m=None, step=1):
  '''range转list'''
  if m is not None:
    return list(range(n, m, step))
  else:
    return list(range(0, n, step))



def transpose(data):
  '''矩阵转置'''
  return map(list, zip(*data))


def windows(iterable, length=2, overlap=0, yield_tail=False):
  '''按照固定窗口大小切片list, 可以重叠
  滑动array窗口,
  每次提供length数目的元素,如果有overlap则重复之前的元素
  yield_tail: 最后不足 length 的那部分元素是否也要 yield'''

  import itertools
  if length <= overlap:
    raise AttributeError('overlap {} cannot larger than length {}'.format(overlap, length))
  it = iter(iterable)
  results = list(itertools.islice(it, length))
  while len(results) == length:
    yield results
    results = results[length-overlap:]
    results.extend(itertools.islice(it, length-overlap))
  if results and yield_tail:
    yield results


def joiner(iterable, sep=', ', group=None, group_sep='\n'):
  '''拼合列表
  如果提供 group, 先按照group长度分组拼合每个group内部元素, 再拼合不同的group'''
  if not group:
    return sep.join(str(elem) for elem in iterable)
  else:
    return group_sep.join(joiner(g, sep) for g in windows(iterable, length=group, yield_tail=True))




def match_previous(lines, pattern, history=5):
  '''返回匹配的行以及之前的n行'''
  from collections import deque
  previous_lines = deque(maxlen=history)
  for li in lines:
    if pattern in li:
      yield li, previous_lines
    previous_lines.append(li)
  # '''usage'''
  # with open(r'../../cookbook/somefile.txt') as f:
  #   for line, prevlines in search(f, 'Python', 5):
  #     for pline in prevlines:
  #       print(pline, end='')
  #     print(line, end='')
  #     print('-' * 20)


def top(iterable, n=1, smallest=False, key=None):
  ''' 返回列表中最大或最小的n个元素
  适合于n比较小的情况
  如果只需要最大或最小的一个元素, 应该用max(), min()
  如果需要非常多的元素, 应该 sorted(iterable)[:n] '''
  import heapq
  if smallest:
    return heapq.nsmallest(n, iterable, key=key)
  else:
    return heapq.nlargest(n, iterable, key=key)
  # portfolio = [
  #     {'name': 'IBM', 'shares': 100, 'price': 91.1},
  #     {'name': 'AAPL', 'shares': 50, 'price': 543.22},
  #     {'name': 'FB', 'shares': 200, 'price': 21.09},
  #     {'name': 'HPQ', 'shares': 35, 'price': 31.75},
  #     {'name': 'YHOO', 'shares': 45, 'price': 16.35},
  #     {'name': 'ACME', 'shares': 75, 'price': 115.65}
  # ]
  # cheap = heapq.nsmallest(3, portfolio, key=lambda s: s['price'])
  # expensive = heapq.nlargest(3, portfolio, key=lambda s: s['price'])








"""--------------
数值操作
--------------"""


def rand(start, end=None):
  if isinstance(start, int):
    if end:
      return random.randint(start, end-1)
    else:
      return random.randint(0, start-1)
  elif isinstance(start, float):
    if end:
      return random.uniform(start, float(end))
    else:
      return random.uniform(0.0, start)


def random_split(seq, p=0.1):
  train = []
  test = []
  roll = random.random
  for row in seq:
    if roll() < p:
      test.append(row)
    else:
      train.append(row)
  return train, test



def statistic(seq, reverse=False, precision=None):
  '''
  统计 list 中的数据
  a = [1,1,1,1,2,2]
  b = a*100 + list(range(1,100))
  c = list('aaaaaaasghkghewxckbv')
  print(statistic(a))
  print(statistic(b))
  print(statistic(c))
  '''

  from collections import Counter
  array = list(seq)
  if precision:
    array = [round(e, precision) for e in array]
  counter = Counter(array)
  length = len(array)
  s = 'statistic ({} items):\n'.format(length)
  s += ''.join('  <{}>: {} ({:.1%})\n'.format(k, v, v/length) for k, v in sorted(list(counter.items()), reverse=reverse))
  if all(isinstance(n, (int, float)) for n in array):
    s += '  sum:{:.4f}  ave:{:.4f}  min:{}  max:{}'.format(sum(array), sum(array)/length, min(array), max(array))
  return s





"""--------------
文件, 文件夹, 文本
--------------"""


def all_files(root, patterns='*', single_level=False, yield_folders=False):
  ''' 取得文件夹下所有文件 '''

  import fnmatch
  patterns = patterns.split(';')
  for path, subdirs, files in os.walk(root):
    if yield_folders:
      files.extend(subdirs)
    files.sort()
    for name in files:
      for pattern in patterns:
        if fnmatch.fnmatch(name, pattern):
          yield os.path.join(path, name)
          break
      if single_level:
        break


def datalines(data, sample=None):
  '''返回一段文字中有效的行(非空行, 且不以注释符号开头)'''
  ret = []
  for l in data.splitlines():
    line = l.strip()
    if line and not line.startswith('#'):
      ret.append(line)
  if sample:
    return ret[:sample]
  else:
    return ret


def datalines_from_file(path):
  '''返回文本文件中有效的行'''
  if not os.path.exists(path):
    raise Exception("not os.path.exists({})".format(path))
  try:
    with open(path, 'r', encoding='utf-8') as f:
      lines = f.read()
  except UnicodeDecodeError:
    with open(path, 'r', encoding='gbk') as f:
      lines = f.read()
  return datalines(lines)


def paragraphs(lines, is_separator=str.isspace, joiner=''.join):
  '''返回文本中的段落'''
  import itertools
  for sep_group, lineiter in itertools.groupby(lines, key=is_separator):
    print(sep_group, list(lineiter))
    if not sep_group:
      yield joiner(lineiter)


def datamatrix(text, title=True, sep=','):
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
    if text.isdigit():
      return int(text)
    try:
      return float(text)
    except ValueError:
      return text

  data = iter(datalines(text))

  regexp = r'\s*{}\s*'.format(sep)
  if title:
    title = re.split(regexp, next(data))
  else:
    title = []

  array = []
  for line in data:
    line = [convert(e) for e in re.split(regexp, line)]
    array.append(line)
  return array, title




def file_timer(path=None, prefix='', suffix='', ext=None):
  '''在文件名中加入时间戳，使其命名唯一'''
  # t = time.strftime('%Y%m%d_%H%M%S')
  t = time.strftime('%m%d_%H%M%S')
  if path is None:
    return prefix+t+suffix
  else:
    path, file = os.path.split(path)
    name, old_ext = os.path.splitext(file)
    if ext:
      old_ext = '.'+ext
    return os.path.join(path, prefix+name+t+suffix+old_ext)


def random_sleep(*arg):
  '''休眠指定的时间,或范围内的随机值'''
  if len(arg) == 1:
    return time.sleep(arg[0])
  else:
    t = random.uniform(float(arg[0]), float(arg[1]))
    return time.sleep(t)





def yaml_ordered_load(stream, Loader=None, object_pairs_hook=None):
  '''按照有序字典载入yaml'''
  import yaml
  from collections import OrderedDict
  if Loader is None:
    Loader = yaml.Loader
  if object_pairs_hook is None:
    object_pairs_hook = OrderedDict

  class OrderedLoader(Loader):
    pass

  def construct_mapping(loader, node):
    loader.flatten_mapping(node)
    return object_pairs_hook(loader.construct_pairs(node))
  OrderedLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, construct_mapping)
  return yaml.load(stream, OrderedLoader)





def _load_csv_value_convert(text):
  '''载入csv时处理每个字段的值, 转换整数等'''
  nan = float('nan')
  if text == 'nan':
    return nan
  if text.isdigit():
    return int(text)
  try:
    return float(text)
  except ValueError:
    return text


def _start_or_end_with(text, pattern):
  pattern = tuple(pattern)
  return text.startswith(pattern) or text.endswith(pattern)

# 这样最终会导致在创建一个命名元组时产生一个 ValueError 异常而失败。
# 为了解决这问题，你可能不得不先去修正列标题。
# 例如，可以像下面这样在非法标识符上使用一个正则表达式替换：

# import re
# with open('stock.csv') as f:
#     f_csv = csv.reader(f)
#     headers = [ re.sub('[^a-zA-Z_]', '_', h) for h in next(f_csv) ]
#     Row = namedtuple('Row', headers)
#     for r in f_csv:
#         row = Row(*r)

# 为了写入CSV数据，你仍然可以使用csv模块，不过这时候先创建一个 writer 对象。例如;

# headers = ['Symbol','Price','Date','Time','Change','Volume']
# rows = [('AA', 39.48, '6/11/2007', '9:36am', -0.18, 181800),
#          ('AIG', 71.38, '6/11/2007', '9:36am', -0.15, 195500),
#          ('AXP', 62.58, '6/11/2007', '9:36am', -0.46, 935000),
#        ]

# with open('stocks.csv','w') as f:
#     f_csv = csv.writer(f)
#     f_csv.writerow(headers)
#     f_csv.writerows(rows)


def load_csv(path, sample=10, only_title=False, include=(), exclude=()):
  from itertools import compress
  import csv

  with open(path) as f:
    titles = f.readline().strip().split(',')

  if include:
    column_compress = [_start_or_end_with(title, include) for title in titles]
  else:
    column_compress = [True] * len(titles)
  if exclude:
    column_compress = [not _start_or_end_with(title, exclude) and tb for title, tb in zip(titles, column_compress)]

  if only_title:
    return list(compress(titles, column_compress))

  with open(path) as f:
    lines = csv.reader(f)
    next(lines)
    result = []
    for i, line in enumerate(lines, 1):
      if sample and i > sample:
        break
      result.append([_load_csv_value_convert(x) for x in compress(line, column_compress)])

  return result




def section_dict(lines, title_pattern='=', reverse=False):
  '''通过小节的标题和之后文字生成 {标题: 内容} 的dict

  title_pattern: 用于识别title的标识符
                 line.startswith(title_pattern) 视为小节标题, 其余视为正文

  reverse: 为 True 时  返回值为 {key: 内容 value: 标题}
           为 False 时 返回值为 {key: 标题 value: 内容}
  '''
  from collections import defaultdict

  title = 'Default'
  result = defaultdict(list)
  for line in datalines(lines):
    if line.startswith(title_pattern):
      title = line
    else:
      if reverse:
        result[line].append(title)
      else:
        result[title].append(line)
  return result










"""--------------
调试和测试
--------------"""
import types


class TestException(Exception):
  pass


def microtest(modulename, verbose=None, log=sys.stdout):
  '''自动测试
  运行模块中所有以 _test_ 开头的没有参数的函数
  modulename = 要测试的模块名称
  verbose    = 打印更多的内容
  成功后返回 None, 失败时报异常'''
  module = __import__(modulename)
  total_tested = 0
  total_failed = 0
  total_pending = 0
  # print(111, file=sys.stdout)
  for name in dir(module):
    # print(name)
    if name.startswith('_test_'):
      obj = getattr(module, name)
      if (isinstance(obj, types.FunctionType) and not obj.__code__.co_argcount):
        if verbose:
          print('        MicroTest: = <%s> on test' % name, file=log)
        try:
          total_tested += 1
          obj()
          print('        MicroTest: . <%s> tested' % name, file=log)
        except Exception:
          total_failed += 1
          print('')
          print('---- ↓ MicroTest: %s.%s() FAILED ↓ ----' % (modulename, name), file=sys.stderr)
          traceback.print_exc()
          print('---- ↑ MicroTest: %s.%s() FAILED ↑ ----' % (modulename, name), file=sys.stderr)
          print('')
    elif 'test' in name:
      total_pending += 1
      if verbose:
        print('        MicroTest: ? <%s> detect pending test' % name, file=log)

  message = '\n\n        MicroTest: module "%s" failed (%s/%s) unittests. (pending %s unittests)\n\n' % (modulename, total_failed, total_tested, total_pending)
  if total_failed > 0:
    raise TestException(message)
  if verbose:
    print(message, file=log)




def create_logger(name=os.path.basename(__file__)):
  import logging
  logger = logging.getLogger(name)
  logger.setLevel(logging.DEBUG)
  # create file handler which logs even debug messages
  fh = logging.FileHandler(name+'.log', encoding='utf-8')
  fh.setLevel(logging.DEBUG)
  # create console handler with a higher log level
  ch = logging.StreamHandler()
  ch.setLevel(logging.DEBUG)
  # create formatter and add it to the handlers
  ch_foramt_str = "%(message)s"
  ch_formatter = logging.Formatter(ch_foramt_str)
  fh_foramt_str = "\n%(message)s\n  -- (%(asctime)s %(name)s %(levelname)s)"
  fh_formatter = logging.Formatter(fh_foramt_str)
  ch.setFormatter(ch_formatter)
  fh.setFormatter(fh_formatter)
  # add the handlers to logger
  logger.addHandler(ch)
  logger.addHandler(fh)
  return logger
  ''' logger.debug('blabla') '''








def watch(variableName, watchOutput=sys.stdout):
  '''调用 watch(secretOfUniverse) 打印出如下的信息：
     # => File "trace.py", line 57, in __testTrace
     # =>   secretOfUniverse <int> = 42'''
  watch_format = ('File "%(fileName)s", line %(lineNumber)d, in'
                  ' %(methodName)s\n  %(varName)s <%(varType)s>'
                  ' = %(value)s\n\n')
  if __debug__:
    stack = traceback.extract_stack()[-2:][0]
    actualCall = stack[3]
    if actualCall is None:
      actualCall = "watch([unknown])"
    left = actualCall.find('(')
    right = actualCall.rfind(')')
    paramDict = dict(varName=actualCall[left+1:right].strip(),
                     varType=str(type(variableName))[8:-2],
                     value=repr(variableName),
                     methodName=stack[2],
                     lineNumber=stack[1],
                     fileName=stack[0])
    watchOutput.write(watch_format % paramDict)






import inspect


class safesub(dict):
  """防止key找不到"""
  def __missing__(self, key):
    return '{missing: ' + key + '}'


def puts(obj=None, file=None, flush=False, trace_line=False, end='\n', sep=', ',
         precision=4, indention=0, max_length=20, ):
  ''' 依据对象类型选择不同的打印形式
  TODO(Probe): 集成 pprint
  提供 file 时 输出到 file 文件中
  否则输出到 stdout

  如果数组很长则可以省略其中部分

  返回值为打印的数据自身

  file: 值为 string 时视为该文件路径, 值为 None 则使用 sys.stdout
  flush: 配合 file 使用, 是否清空已存在的文件内容
  trace_line: 是否打印调用 puts 的脚本名称和所在行
  end: 全部打印结束后的结束字符
  sep: 打印 list 和 tuple 元素的间隔符
  precision: 控制 float 的显示精度
  indention: 控制字典的打印缩进, 同时也指示是否为首次调用puts
  max_length: 打印 list 和 tuple 时最多打印的元素数量


  形如 'Hello {name}' 的字符串允许变量缺失

  有两种插值方法:

  1. str.format() 风格的
     >>> puts('result is {result}, bm is {bm}')

  2. 变量名紧挨着等号的
     >>> puts('result=, bm=')
     >>> puts('result.len=, bm=')

  '''
  # 同时指定宽度和精度的一般形式是 '[<>^]?width[,]?(.digits)?'
  # 其中 width 和 digits 为整数 ? 代表可选部分
  # 同样的格式也被用在字符串的 format() 方法中 比如：
  # >>> 'The value is {:0,.2f}'.format(x)
  # 'The value is 1,234.57'

  should_close_file = False
  if file is None:
    file = sys.stdout
  elif isinstance(file, str):
    file = open(file, ('w' if flush else 'a'), encoding='utf8')
    should_close_file = True



  if obj is None:
    print('\n------------------', end=end, file=file)

  elif isinstance(obj, int):
    print(obj, end=end, file=file)

  elif isinstance(obj, float):
    print(round(obj, precision), end=end, file=file)

  elif isinstance(obj, str):
    if indention is 0:  # 首次调用, 需处理可能嵌入的{}变量
      indention += 1
      if '{' in obj or '=' in obj:  # string里有包含的变量
        # puts('result=, bm=')
        # puts('result.len=, bm=')
        # puts('result={result}, bm={bm}')
        obj = re.sub(r'\b([A-Za-z_][a-zA-Z_0-9.]*)(?:=)', r'\1={\1}', obj)
        vars_safemap = safesub(sys._getframe(1).f_locals)

        var = r'(?P<var>\{[a-zA-Z_][a-zA-Z_0-9]*\})'
        var_method = r'(?P<var_method>\{[a-zA-Z_][a-zA-Z_0-9]*\.\w+?\})'
        normal_text = r'(?P<normal_text>[^{}]*)'
        master_pattern = re.compile('|'.join([var, var_method, normal_text]))

        for m in iter(master_pattern.scanner(obj).match, None):
          token, value = m.lastgroup, m.group()
          if not value:
            continue
          if token == 'normal_text':
            print(value, end='', file=file)
          elif token == 'var':  # 形如 '{result}'
            puts(vars_safemap[value[1:-1]], file=file, flush=flush, end='', sep=sep,
                 precision=precision, indention=indention, max_length=max_length)
          elif token == 'var_method':  # 形如 '{result.len}'
            methods = {'len': len, 'any': any, 'all': all}
            part1, part2 = value[1:-1].split('.')
            puts(methods[part2].__call__(vars_safemap[part1]),
                 file=file, flush=flush, end='', sep=sep,
                 precision=precision, indention=indention, max_length=max_length)

        if trace_line:
          stack = traceback.extract_stack()[-2:][0]
          # script_name=stack[0]
          # line_number=stack[1]
          # method_name=stack[2]
          print('\n    File "{0}", line {1}'.format(*stack), end=end, file=file)
        else:
          print('', end=end, file=file)
      else:  # 没有特殊字符, 直接打印字符串
        print(obj, end=end, file=file)
    else:    # 并非首次调用, 直接打印字符串, 不需要sub嵌入的变量 {result}
      print(obj, end=end, file=file)


  elif isinstance(obj, (list, tuple)):
    array_obj = obj
    array_length = len(array_obj)
    print('[', end='', file=file)
    if array_length > max_length:
      tail_length = int(max_length * 0.1) + 1
      head_length = max_length - tail_length
      trimed = '...trimed {} items...'.format(array_length - max_length)
      array_obj = array_obj[:head_length] + [trimed] + array_obj[-tail_length:]

    for i, elem in enumerate(array_obj):
      elem_end = '' if i+1 == len(array_obj) else sep
      puts(elem, file=file, end=elem_end, sep=sep,
           precision=precision, max_length=max_length)

    if array_length > 10:
      print('] <--length {}'.format(array_length), end=end, file=file)
    else:
      print(']', end=end, file=file)

  elif isinstance(obj, dict):
    padding = '  ' * indention
    print('{', end='\n', file=file)
    indention += 1
    for i, (k, v) in enumerate(obj.items(), 1):
      print('  ' * indention + str(k) + ': ', end='', file=file)
      puts(v, file=file, flush=flush, end='\n', sep=sep,
           precision=precision, indention=indention, max_length=max_length)
    print(padding + '} <-- '+str(i)+' items', end='\n', file=file)


  # is{module|class|function|method|builtin}(obj): 检查对象是否为模块 类 函数 方法 内建函数或方法
  elif inspect.ismodule(obj):
    s = 'module {} __dict__ ({})\n'.format(obj.__name__, len(obj.__dict__))
    items = []
    for x, v in obj.__dict__.items():
      v = str(v)
      if len(str(v)) > 60:
        v = str(v)[:60] + '...'
      items.append('{}:    {}'.format(x, v))
    print(s + '\n'.join(items), end=end, file=file)

  elif inspect.isclass(obj):
    s = 'class {} __dict__ ({})\n'.format(obj.__name__, len(obj.__dict__))
    s += str(obj.__dict__) + '\n'
    s += str(dir(obj))
    print(s, end=end, file=file)


  elif inspect.isfunction(obj):
    print('function: '+obj.__name__+' '+str(inspect.getargspec(obj)), end=end, file=file)
    print(obj.__doc__, end=end, file=file)
    # Notes Python 2  Python 3
    # a_function.func_name      a_function.__name__
    # a_function.func_doc       a_function.__doc__
    # a_function.func_defaults  a_function.__defaults__
    # a_function.func_dict      a_function.__dict__
    # a_function.func_closure   a_function.__closure__
    # a_function.func_globals   a_function.__globals__
    # a_function.func_code      a_function.__code__
  elif inspect.ismethod(obj):
    s = 'method of '+str(obj.__self__)+': '
    s += obj.__name__+' '+str(inspect.getargspec(obj))
    s += '\n{}'.format(obj.__doc__)
    print(s, end=end, file=file)
    # 在Python 2里，类方法可以访问到定义他们的类对象(class object)，
    # 也能访问方法对象(method object)本身。
    # im_self是类的实例对象；im_func是函数对象，im_class是类本身。
    # 在Python 3里，这些属性被重新命名，以遵循其他属性的命名约定。
    # Notes Python 2  Python 3
    # aClassInstance.aClassMethod.im_func   aClassInstance.aClassMethod.__func__
    # aClassInstance.aClassMethod.im_self   aClassInstance.aClassMethod.__self__
    # aClassInstance.aClassMethod.im_class  aClassInstance.aClassMethod.__self__.__class__

  else:  # an instance??
    s = '{} ({})\n'.format(obj.__class__, len(dir(obj)))
    s += '    '.join(x for x in dir(obj))
    print(s, end=end, file=file)



  if should_close_file:
    file.close()
  return obj

  # end of puts







































"""--------------
对象和实例方法
--------------"""


def nested_property(c):
  ''' 嵌套 @property
  可以用一个 def 解决问题
  使用方法
  import math
  class Rectangle(object):
    def __init__(self, x, y):
      self.x = x
      self.y = y
    @nested_property
    def area():
      doc = "Area of the rectangle"
      def fget(self):
        return self.x * self.y
      def fset(self, value):
        ratio = math.sqrt((1.0 * value) / self.area)
        self.x *= ratio
        self.y *= ratio
      return locals()
  '''
  return property(**c())


class Singleton(object):
  """单例模式
  继承这个类之后只能创建一个实例"""
  def __new__(cls, *args, **kwargs):
    if '_inst' not in vars(cls):
      # print(super(Singleton, cls))
      cls._inst = super(Singleton, cls).__new__(cls)
      # cls._inst = super(Singleton, cls).__new__(cls, *args, **kwargs)
    return cls._inst


class AutoDelegator(object):
  ''' 自动代理
  class Pricing(AutoDelegator):
    def __init__(self, location, event):
      self.delegates = [location, event]
    def setlocation(self, location):
      self.delegates[0] = location

  '''
  delegates = ()
  do_not_delegate = ()

  def __getattr__(self, key):
    if key not in self.do_not_delegate:
      for d in self.delegates:
        try:
          return getattr(d, key)
        except AttributeError:
          pass
    raise AttributeError(key)








"""--------------
其他
--------------"""
if "win32" in sys.platform or "win64" in sys.platform:
  import win32clipboard
  import win32con

  def clipboard_get_text():
    ''' 使用Windows剪切板 '''
    win32clipboard.OpenClipboard()
    d = win32clipboard.GetClipboardData(win32con.CF_TEXT)
    win32clipboard.CloseClipboard()
    return d

  def clipboard_set_text(text):
    win32clipboard.OpenClipboard()
    win32clipboard.EmptyClipboard()
    win32clipboard.SetClipboardText(text)
    win32clipboard.CloseClipboard()






























'''
from collections import namedtuple
# >>> from collections import namedtuple
# >>> Subscriber = namedtuple('Subscriber', ['addr', 'joined'])
# >>> sub = Subscriber('jonesy@example.com', '2012-10-19')
# >>> sub
# Subscriber(addr='jonesy@example.com', joined='2012-10-19')
# >>> sub.addr
# 'jonesy@example.com'
# >>> sub.joined
# '2012-10-19'
'''






'''
random.random()
生成一个随机浮点数：range[0.0,1.0)
random.uniform(a,b)
生成一个指定范围内的随机浮点数，a,b为上下限，只要a!=b,就会生成介于两者之间的一个浮点数，若a=b，则生成的浮点数就是a
random.randint(a,b)
生成一个指定范围内的整数，a为下限，b为上限，生成的随机整数a<=n<=b;若a=b，则n=a；若a>b，报错
random.randrange([start], stop [,step])
从指定范围内，按指定基数递增的集合中获取一个随机数，基数缺省值为1
random.randrange(10,100)
输出为10到100间的任意数
random.randrange(10,100,4)
输出为10到100内以4递增的序列[10,14,18,22...]
random.choice(range(10,100,4))
输出在结果上与上一条等效
random.choice(sequence)
从序列中获取一个随机元素，参数sequence表示一个有序类型，泛指list，tuple，字符串等
random.shuffle(x[,random])
将一个列表中的元素打乱
random.sample(sequence,k)
随机取k个元素作为片段返回，不会修改原有序列
'''



# pattern = re.compile(r'hello')
# match = pattern.match('hello world!')
# if match:
#     print match.group()

# re.compile(strPattern[, flag]):
'''
flag
匹配模式，取值可以使用按位或运算符'|'表示同时生效，比如re.I | re.M。
re.compile('pattern', re.I | re.M)与re.compile('(?im)pattern')是等价的。
re.I(re.IGNORECASE): 忽略大小写（括号内是完整写法，下同）
M(MULTILINE): 多行模式，改变'^'和'$'的行为（参见上图）
S(DOTALL): 点任意匹配模式，改变'.'的行为
L(LOCALE): 使预定字符类 \w \W \b \B \s \S 取决于当前区域设定
U(UNICODE): 使预定字符类 \w \W \b \B \s \S \d \D 取决于unicode定义的字符属性
X(VERBOSE): 详细模式。这个模式可以是多行，忽略空白字符，并可以加入注释。
'''
# m = re.match(r'hello', 'hello world!')
# print m.group()


'''
Match对象

匹配的结果，包含了很多关于此次匹配的信息，
可以使用Match提供的可读属性或方法来获取这些信息。
属性：
string: 匹配时使用的文本。
re: 匹配时使用的Pattern对象。
pos: 文本中正则表达式开始搜索的索引。值与Pattern.match()和Pattern.seach()方法的同名参数相同。
endpos: 文本中正则表达式结束搜索的索引。值与Pattern.match()和Pattern.seach()方法的同名参数相同。
lastindex: 最后一个被捕获的分组在文本中的索引。如果没有被捕获的分组，将为None。
lastgroup: 最后一个被捕获的分组的别名。如果这个分组没有别名或者没有被捕获的分组，将为None。
'''

'''
方法：
group([group1, …]):
获得一个或多个分组截获的字符串；指定多个参数时将以元组形式返回。
group1可以使用编号也可以使用别名；编号表整个匹配的子串；
不填写参数时，返回group(0)；
截获字符串的组返回None；
截获了多次的组返回最后一次截获的子串。
groups([default]):
以元组形式返回全部分组截获的字符串。相当于调用group(1,2,…last)。
ult表示没有截获字符串的组以这个值替代，默认为None。
groupdict([default]):
返回以有别名的组的别名为键、以该组截获的子串为值的字典，
没有别名的组不包含在内。default含义同上。
start([group]):
返回指定的组截获的子串在string中的起始索引（子串第一个字符的索引）。group默认值为0。
end([group]):
返回指定的组截获的子串在string中的结束索引（子串最后一个字符的索引+1）。group默认值为0。
span([group]):
返回(start(group), end(group))。
expand(template):
将匹配到的分组代入template中然后返回。
template中可以使用\id或\g<id>、\g<name>引用分组，
但不能使用编号0。
\id与\g<id>是等价的；但\10将被认为是第10个分组，
如果你想表达\1之后是字符'0'，只能使用\g<1>0。
'''
# m = re.match(r'(\w+) (\w+)(?P<sign>.*)', 'hello world!')
# print("m.string:", m.string)                          # m.string: hello world!
# print("m.re:", m.re)                                  # m.re: <_sre.SRE_Pattern object at 0x016E1A38>
# print("m.pos:", m.pos)                                # m.pos: 0
# print("m.endpos:", m.endpos)                          # m.endpos: 12
# print("m.lastindex:", m.lastindex)                    # m.lastindex: 3
# print("m.lastgroup:", m.lastgroup)                    # m.lastgroup: sign
# print("m.group(1,2):", m.group(1, 2))                 # m.group(1,2): ('hello', 'world')
# print("m.groups():", m.groups())                      # m.groups(): ('hello', 'world', '!')
# print("m.groupdict():", m.groupdict())                # m.groupdict(): {'sign': '!'}
# print("m.start(2):", m.start(2))                      # m.start(2): 6
# print("m.end(2):", m.end(2))                          # m.end(2): 11
# print("m.span(2):", m.span(2))                        # m.span(2): (6, 11)
# print(r"m.expand(r'\2 \1\3'):", m.expand(r'\2 \1\3')) # m.expand(r'\2 \1\3'): world hello!



'''
Pattern对象
是一个编译好的正则表达式，通过Pattern提供的一系列方法可以对文本进行匹配查找。
Pattern不能直接实例化，必须使用re.compile()进行构造。
pattern: 编译时用的表达式字符串。
flags: 编译时用的匹配模式。数字形式。
groups: 表达式中分组的数量。
groupindex: 以表达式中有别名的组的别名为键、以该组对应的编号为值的字典，没有别名的组不包含在内。
'''
# p = re.compile(r'(\w+) (\w+)(?P<sign>.*)', re.DOTALL)
# print("p.pattern:", p.pattern)       # p.pattern: (\w+) (\w+)(?P<sign>.*)
# print("p.flags:", p.flags)           # p.flags: 48
# print("p.groups:", p.groups)         # p.groups: 3
# print("p.groupindex:", p.groupindex) # p.groupindex: {'sign': 3}

'''
match(string[, pos[, endpos]]) | re.match(pattern, string[, flags]):
将从string的pos下标处起尝试匹配pattern；
如果pattern结束时仍可匹配，则返回一个Match对象；
如果匹配过程中pattern无法匹配，或者匹配未结束就已到达endpos，则返回None。

pos和endpos的默认值分别为0和len(string)；
re.match()无法指定这两个参数，参数flags用于编译pattern时指定匹配模式。

注意：这个方法并不是完全匹配。当pattern结束时若string还有剩余字符，仍然视为成功。
想要完全匹配，可以在表达式末尾加上边界匹配符'$'。
'''

'''
search(string[, pos[, endpos]]) | re.search(pattern, string[, flags]):
用于查找字符串中可以匹配成功的子串。
从string的pos下标处起尝试匹配pattern，
如果pattern结束时仍可匹配，则返回一个Match对象；
若无法匹配，则将pos加1后重新尝试匹配；
直到pos=endpos时仍无法匹配则返回None。
pos和endpos的默认值分别为0和len(string))；
re.search()无法指定这两个参数，参数flags用于编译pattern时指定匹配模式。
'''
# pattern = re.compile(r'world')
# # 这个例子中使用match()无法成功匹配
# match = pattern.search('hello world!')
# if match:
#   print match.group()




'''
split(string[, maxsplit]) | re.split(pattern, string[, maxsplit]):
按照能够匹配的子串将string分割后返回列表。
maxsplit用于指定最大分割次数，不指定将全部分割。
'''
# p = re.compile(r'\d+')
# print(p.split('one1two2three3four4')) # ['one', 'two', 'three', 'four', '']
'''
findall(string[, pos[, endpos]]) | re.findall(pattern, string[, flags]):
搜索string，以列表形式返回全部能匹配的子串。
'''
# p = re.compile(r'\d+')
# print(p.findall('one1two2three3four4')) # ['1', '2', '3', '4']
'''
finditer(string[, pos[, endpos]]) | re.finditer(pattern, string[, flags]):
搜索string，返回一个顺序访问每一个匹配结果（Match对象）的迭代器。
'''
# p = re.compile(r'\d+')
# for m in p.finditer('one1two2three3four4'):
#   print(m.group()) # 1 2 3 4


'''
sub(repl, string[, count]) | re.sub(pattern, repl, string[, count]):
使用repl替换string中每一个匹配的子串后返回替换后的字符串。
当repl是字符串时，可以使用\id或\g<id>、\g<name>引用分组，但不能使用编号0。
当repl是方法时，这个方法应当只接受一个参数（Match对象），
并返回一个字符串用于替换（返回的字符串中不能再引用分组）。
count用于指定最多替换次数，不指定时全部替换。
'''
# p = re.compile(r'(\w+) (\w+)')
# s = 'i say, hello world!'
# print(p.sub(r'\2 \1', s)) # say i, world hello!
# def func(m):
#   return m.group(1).title() + ' ' + m.group(2).title()
# print(p.sub(func, s)) # I Say, Hello World!

'''
subn(repl, string[, count]) | re.sub(pattern, repl, string[, count]):
返回 (sub(repl, string[, count]), 替换次数)。
'''
# p = re.compile(r'(\w+) (\w+)')
# s = 'i say, hello world!'
# print(p.subn(r'\2 \1', s)) # ('say i, world hello!', 2)
# def func(m):
#   return m.group(1).title() + ' ' + m.group(2).title()
# print(p.subn(func, s)) # ('I Say, Hello World!', 2)





'''
Python常见文件操作示例

os.path 模块中的路径名访问函数

分隔
os.path.basename() 去掉目录路径, 返回文件名
os.path.dirname() 去掉文件名, 返回目录路径
os.path.join() 将分离的各部分组合成一个路径名
os.path.split() 返回 (dirname(), basename()) 元组
os.path.splitdrive() 返回 (drivename, pathname) 元组
os.path.splitext() 返回 (filename, extension) 元组

信息
os.path.getatime() 返回最近访问时间
os.path.getctime() 返回文件创建时间
os.path.getmtime() 返回最近文件修改时间
os.path.getsize() 返回文件大小(以字节为单位)

查询
os.path.exists() 指定路径(文件或目录)是否存在
os.path.isabs() 指定路径是否为绝对路径
os.path.isdir() 指定路径是否存在且为一个目录
os.path.isfile() 指定路径是否存在且为一个文件
os.path.islink() 指定路径是否存在且为一个符号链接
os.path.ismount() 指定路径是否存在且为一个挂载点
os.path.samefile() 两个路径名是否指向同个文件

os.path.abspath(name):获得绝对路径
os.path.normpath(path):规范path字符串形式

分离文件名：os.path.split(r"c:\python\hello.py") --> ("c:\\python", "hello.py")
分离扩展名：os.path.splitext(r"c:\python\hello.py") --> ("c:\\python\\hello", ".py")
获取路径名：os.path.dirname(r"c:\python\hello.py") --> "c:\\python"
获取文件名：os.path.basename(r"r:\python\hello.py") --> "hello.py"
判断文件是否存在：os.path.exists(r"c:\python\hello.py") --> True
判断是否是绝对路径：os.path.isabs(r".\python\") --> False
判断是否是目录：os.path.isdir(r"c:\python") --> True
判断是否是文件：os.path.isfile(r"c:\python\hello.py") --> True
判断是否是链接文件：os.path.islink(r"c:\python\hello.py") --> False
获取文件大小：os.path.getsize(filename)
搜索目录下的所有文件：os.path.walk()

'''


'''
os模块中的文件操作：
os 模块属性
linesep 用于在文件中分隔行的字符串
sep 用来分隔文件路径名的字符串
pathsep 用于分隔文件路径的字符串
curdir 当前工作目录的字符串名称
pardir (当前工作目录的)父目录字符串名称

重命名：os.rename(old, new)
删除：os.remove(file)
列出目录下的文件：os.listdir(path)
获取当前工作目录：os.getcwd()
改变工作目录：os.chdir(newdir)
创建多级目录：os.makedirs(r"c:\python\test")
创建单个目录：os.mkdir("test")
删除多个目录：os.removedirs(r"c:\python") #删除所给路径最后一个目录下所有空目录。
删除单个目录：os.rmdir("test")
获取文件属性：os.stat(file)
修改文件权限与时间戳：os.chmod(file)
执行操作系统命令：os.system("dir")
启动新进程：os.exec(), os.execvp()
在后台执行程序：osspawnv()
终止当前进程：os.exit(), os._exit()
'''




'''
shutil模块对文件的操作：
复制单个文件：shutil.copy(oldfile, newfile)
复制整个目录树：shutil.copytree(r".\setup", r".\backup")
删除整个目录树：shutil.rmtree(r".\backup")
临时文件的操作：
创建一个唯一的临时文件：tempfile.mktemp() --> filename
打开临时文件：tempfile.TemporaryFile()
内存文件（StringIO和cStringIO）操作
[4.StringIO] #cStringIO是StringIO模块的快速实现模块

创建内存文件并写入初始数据：f = StringIO.StringIO("Hello world!")
读入内存文件数据：print f.read() #或print f.getvalue() --> Hello world!
想内存文件写入数据：f.write("Good day!")
关闭内存文件：f.close()

'''



'''
最常用的time.time()返回的是一个浮点数，单位为秒。
但strftime处理的类型是time.struct_time，
实际上是一个tuple。
strptime和localtime都会返回这个类型。

>>> import time
>>> t = time.time()
>>> t
1202872416.4920001
>>> type(t)
<type 'float'>
>>> t = time.localtime()
>>> t
(2008, 2, 13, 10, 56, 44, 2, 44, 0)
>>> type(t)
<type 'time.struct_time'>
>>> time.strftime('%Y-%m-%d', t)
'2008-02-13'
>>> time.strptime('2008-02-14', '%Y-%m-%d')
(2008, 2, 14, 0, 0, 0, 3, 45, -1)
'''



'''
>>> import string
>>> string.ascii_letters
'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
>>> string.ascii_lowercase
'abcdefghijklmnopqrstuvwxyz'
>>> string.ascii_uppercase
'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
>>> string.digits
'0123456789'
>>> string.hexdigits
'0123456789abcdefABCDEF'
>>> string.letters
'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
>>> string.lowercase
'abcdefghijklmnopqrstuvwxyz'
>>> string.uppercase
'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
>>> string.octdigits
'01234567'
>>> string.punctuation
'!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
>>> string.printable
'0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~ \t\n\r\x0b\x0c'
>>> string.whitespace
'\t\n\x0b\x0c\r


>>> "hello".capitalize()
'Hello'
>>> "hello world".capitalize()
'Hello world'
>>> "              adsd         ".strip()
'adsd'
>>> "              adsd         ".rstrip()
'              adsd'
>>> "              adsd         ".lstrip()
'adsd         '

>>> "Helloo".swapcase()
'hELLOO'

>>> 'ww'.ljust(20)
'ww                  '
>>> 'ww'.rjust(20)
'                  ww'
>>> 'ww'.center(20)
'         ww         '

>>> 'ww'.zfill(20)
Pad a numeric string on the left with zero digits until the given width is reached. Strings starting with a sign are handled correctly.
'000000000000000000ww'

'''


# str.format
'''
>>> '{0}, {1}, {2}'.format('a', 'b', 'c')
'a, b, c'
>>> '{}, {}, {}'.format('a', 'b', 'c')  # 2.7+ only
'a, b, c'
>>> '{2}, {1}, {0}'.format('a', 'b', 'c')
'c, b, a'
>>> '{2}, {1}, {0}'.format(*'abc')      # unpacking argument sequence
'c, b, a'
>>> '{0}{1}{0}'.format('abra', 'cad')   # arguments' indices can be repeated
'abracadabra'

>>> 'Coordinates: {latitude}, {longitude}'.format(latitude='37.24N', longitude='-115.81W')
'Coordinates: 37.24N, -115.81W'
>>> coord = {'latitude': '37.24N', 'longitude': '-115.81W'}
>>> 'Coordinates: {latitude}, {longitude}'.format(**coord)
'Coordinates: 37.24N, -115.81W'

>>> c = 3-5j
>>> ('The complex number {0} is formed from the real part {0.real} '
...  'and the imaginary part {0.imag}.').format(c)
'The complex number (3-5j) is formed from the real part 3.0 and the imaginary part -5.0.'
>>> class Point(object):
...     def __init__(self, x, y):
...         self.x, self.y = x, y
...     def __str__(self):
...         return 'Point({self.x}, {self.y})'.format(self=self)
...
>>> str(Point(4, 2))
'Point(4, 2)

>>> coord = (3, 5)
>>> 'X: {0[0]};  Y: {0[1]}'.format(coord)
'X: 3;  Y: 5'

>>> "repr() shows quotes: {!r}; str() doesn't: {!s}".format('test1', 'test2')
"repr() shows quotes: 'test1'; str() doesn't: test2"

>>> '{:<30}'.format('left aligned')
'left aligned                  '
>>> '{:>30}'.format('right aligned')
'                 right aligned'
>>> '{:^30}'.format('centered')
'           centered           '
>>> '{:*^30}'.format('centered')  # use '*' as a fill char
'***********centered***********'

>>> '{:+f}; {:+f}'.format(3.14, -3.14)  # show it always
'+3.140000; -3.140000'
>>> '{: f}; {: f}'.format(3.14, -3.14)  # show a space for positive numbers
' 3.140000; -3.140000'
>>> '{:-f}; {:-f}'.format(3.14, -3.14)  # show only the minus -- same as '{:f}; {:f}'
'3.140000; -3.140000'

>>> # format also supports binary numbers
>>> "int: {0:d};  hex: {0:x};  oct: {0:o};  bin: {0:b}".format(42)
'int: 42;  hex: 2a;  oct: 52;  bin: 101010'
>>> # with 0x, 0o, or 0b as prefix:
>>> "int: {0:d};  hex: {0:#x};  oct: {0:#o};  bin: {0:#b}".format(42)
'int: 42;  hex: 0x2a;  oct: 0o52;  bin: 0b101010'

>>> '{:,}'.format(1234567890)
'1,234,567,890'

>>> points = 19.5
>>> total = 22
>>> 'Correct answers: {:.2%}.'.format(points/total)
'Correct answers: 88.64%'

>>> import datetime
>>> d = datetime.datetime(2010, 7, 4, 12, 15, 58)
>>> '{:%Y-%m-%d %H:%M:%S}'.format(d)
'2010-07-04 12:15:58'

>>> for align, text in zip('<^>', ['left', 'center', 'right']):
...     '{0:{fill}{align}16}'.format(text, fill=align, align=align)
...
'left<<<<<<<<<<<<'
'^^^^^center^^^^^'
'>>>>>>>>>>>right'
>>>
>>> octets = [192, 168, 0, 1]
>>> '{:02X}{:02X}{:02X}{:02X}'.format(*octets)
'C0A80001'
>>> int(_, 16)
3232235521
>>>
>>> width = 5
>>> for num in range(5,12):
...     for base in 'dXob':
...         print '{0:{width}{base}}'.format(num, base=base, width=width),
...     print
...
    5     5     5   101
    6     6     6   110
    7     7     7   111
    8     8    10  1000
    9     9    11  1001
   10     A    12  1010
   11     B    13  1011

'''







# HTML特殊字符

# 后面的两列编号，第一列是用于HTML的，需要在前面加上&#符号；
# 第二列用于CSS文件，但需要用反斜杠\转义；也可用于JavaScript，需要用\u来转义。

# 各种箭头

# ← 8592 2190
# → 8594 2192
# ↑ 8593 2191
# ↓ 8595 2193
# 基本形状
# ▲ 9650 25B2
# ▼ 9660 25BC
# ★ 9733 2605
# ☆ 9734 2606
# ◆ 9670 25C6
# 标点
# « 171 00AB
# » 187 00BB
# ‹ 139 008B
# › 155 009B
# “ 8220 201C
# ” 8221 201D
# ‘ 8216 2018
# ’ 8217 2019
# • 8226 2022
# ◦ 9702 25E6
# ¡ 161 00A1
# ¿ 191 00BF
# ℅ 8453 2105
# № 8470 2116
# & 38 0026
# @ 64 0040
# ℞ 8478 211E
# ℃ 8451 2103
# ℉ 8457 2109
# ° 176 00B0
# | 124 007C
# ¦ 166 00A6
# – 8211 2013
# — 8212 2014
# … 8230 2026
# ¶ 182 00B6
# ∼ 8764 223C
# ≠ 8800 2260
# 法律符号
# ® 174 00AE
# © 169 00A9
# ℗ 8471 2117
# ™ 153 0099
# 货币
# $ 36 0024
# ¢ 162 00A2
# £ 163 00A3
# ¤ 164 00A4
# € 8364 20AC
# ¥ 165 00A5










if __name__ == '__main__':
  # thefiles = list(all_files('/tmp', '*.py;*.htm;*.html'))
  # for path in all_files('/tmp', '*.py;*.htm;*.html'):
  #   print(path)
  # info(os)
  # info(sys)
  # a = list(range(1, 30))
  # print(joiner(a, limit=5))
  # a = list(range(30))
  # print(joiner(a, limit=None))


  arg = 1000.12345678912345678

  puts('arg=')

  # a = 100
  # b = -30.2
  # bb = -30.2222222222222

  # text = 'test tes.101= hehe bb='
  # text2 = 'test tes.101= hehe bb='

  # c = 'hehe'
  # d = all
  # e = list(range(1, 10))
  # e300 = list(range(1, 300))
  # e101 = list(range(1, 102))
  # e12 = list(range(1, 13))

  # puts(a)
  # puts(b, end='...')
  # puts()
  # puts(bb)
  # puts(text)

  # puts('test{a} {bb}')

  # puts('a=')
  # puts('a= b=')

  # puts(c)
  # puts(e)
  # puts(e, sep='--')

  # puts(e300, sep=' ')
  # puts(e12, sep='--')
  # puts(e101, end='heheh')
  # puts(e101, path='test_out.txt')

  # puts('array e300=')


  # puts(all)
  # class Text:
  #   def __init__(self, a):
  #     self.a = a

  #   def call_method(self, b):
  #     '''doc'''
  #     return b

  # text = Text(111)
  # d1 = {a:1, b:2, c:'hehehhahah'}
  # d2 = {'d1': d1, 'text': text}
  # puts(d1)
  # puts()
  # puts(d2)
  # import collections
  # d3 = collections.Counter(list(range(60)))
  # d1['Counter'] = d3
  # puts()
  # puts(d2)



  # puts(text, end='\n\n')
  # puts(Text, end='\n\n')
  # puts(text.call_method)















  # puts(file_timer(path='dsghkl.pdf'))
  # puts(file_timer(path='D:/gsg/dsghkl.pdf'))
  # puts(file_timer(path='D:\\gsg\\dsghkl.pdf'))
  # puts(file_timer(prefix='2012_'))
  # puts(file_timer(suffix='_hehe'))


  # import time
  # t = time.strftime('.backup-%Y-%m-%d-%H%M%S')
  # # info(t)

  # from collections import Counter
  # randoms = [rand(10) for i in range(100000)]
  # c = Counter(randoms)
  # print(sorted(c.keys()))
  # print(c)

  # randoms = [rand(10,12) for i in range(100000)]
  # c = Counter(randoms)
  # print(sorted(c.keys()))
  # print(c)

  # randoms = [rand(2.0) for i in range(100000)]
  # print(min(randoms), max(randoms))