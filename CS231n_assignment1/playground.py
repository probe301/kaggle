
# import random
# import numpy as np
# from cs231n.data_utils import load_CIFAR10
# import matplotlib.pyplot as plt
# a = 100
# data = np.array([float('nan')] * a)

import json
from pprint import pprint

with open('doodles_demo.json') as json_data:
    d = json.loads(json_data.read())
    json_data.close()
    pprint(d)
