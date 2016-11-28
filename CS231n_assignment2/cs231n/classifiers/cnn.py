import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


class ThreeLayerConvNet(object):
  """
  A three-layer convolutional network with the following architecture:

  conv - relu - 2x2 max pool - affine - relu - affine - softmax

  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """

  def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
               hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
               dtype=np.float32):
    """

    Initialize a new network.

    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.params = {}
    self.reg = reg
    self.dtype = dtype

    ############################################################################
    # TODO: Initialize weights and biases for the three-layer convolutional    #
    # network. Weights should be initialized from a Gaussian with standard     #
    # deviation equal to weight_scale; biases should be initialized to zero.   #
    # All weights and biases should be stored in the dictionary self.params.   #
    # Store weights and biases for the convolutional layer using the keys 'W1' #
    # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
    # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
    # of the output affine layer.                                              #
    ############################################################################

    dist_normal = lambda size: np.random.normal(loc=0, scale=weight_scale, size=size)
    dist_zero = lambda size: np.zeros(size)

    C, H, W = input_dim
    F = num_filters
    HH = WW = filter_size
    Hout, Wout = int(H / 2), int(W / 2)

    self.params['W1'] = dist_normal((F, C, HH, WW))
    self.params['b1'] = dist_zero(F)
    self.params['W2'] = dist_normal((F*Hout*Wout, hidden_dim))
    self.params['b2'] = dist_zero(hidden_dim)
    self.params['W3'] = dist_normal((hidden_dim, num_classes))
    self.params['b3'] = dist_zero(num_classes)

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    for k, v in self.params.items():
      self.params[k] = v.astype(dtype)


  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.

    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']

    # pass conv_param to the forward pass for the convolutional layer
    filter_size = W1.shape[2]
    conv_param = {'stride': 1, 'pad': int((filter_size - 1) / 2)}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the three-layer convolutional net,  #
    # computing the class scores for X and storing them in the scores          #
    # variable.                                                                #
    ############################################################################
    pass
    # conv - relu - 2x2 max pool - affine - relu - affine - softmax
    # conv_relu_pool_backward(dout, cache) / dx, dw, db
    N, C, H, W = X.shape
    crp_out, crp_cache = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)
    N, F, Hout, Wout = crp_out.shape
    crp_out = crp_out.reshape((N, -1))  # N, F, Hout, Wout -> N, F*Hout*Wout
    hidden_out, hidden_cache = affine_relu_forward(crp_out, W2, b2)
    scores, scores_cache = affine_forward(hidden_out, W3, b3)


    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    if y is None:
      return scores

    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the three-layer convolutional net, #
    # storing the loss and gradients in the loss and grads variables. Compute  #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    ############################################################################
    pass
    scores -= np.max(scores, axis=1).reshape(N, 1)
    Sum_e_scores = np.sum(np.exp(scores), axis=1).reshape(N, 1)
    Probs = np.exp(scores) / Sum_e_scores
    loss = np.sum(-np.log(Probs[np.arange(N), y]))

    loss /= N
    loss += 0.5 * self.reg * np.sum(W1 * W1)
    loss += 0.5 * self.reg * np.sum(W2 * W2)
    loss += 0.5 * self.reg * np.sum(W3 * W3)

    # end of loss, start grad
    Probs[np.arange(N), y] -= 1
    Probs /= N    # part dL/df
    dout, dw, db = affine_backward(Probs, scores_cache)
    grads['W3'] = dw + self.reg * W3
    grads['b3'] = db

    dout, dw, db = affine_relu_backward(dout, hidden_cache)
    grads['W2'] = dw + self.reg * W2
    grads['b2'] = db

    # 对应 crp_out = crp_out.reshape((N, -1))
    # N, F, Hout, Wout -> N, F*Hout*Wout
    dout = dout.reshape((N, F, Hout, Wout))
    dx, dw, db = conv_relu_pool_backward(dout, crp_cache)
    grads['W1'] = dw + self.reg * W1
    grads['b1'] = db

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return loss, grads












































def print_shape(x):
  import sys
  local = sys._getframe(1).f_locals
  print(x+'.shape=', local.get(x).shape)






class CustomConvNet(object):
  """
  [conv-relu-pool(2,2)]xN - [affine-relu]xM - affine - [softmax]
  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """
  def __init__(self, input_dim=(3, 32, 32),
               filters=(32, 30),
               filter_size=7,
               hidden_dims=(100, 85),
               num_classes=10,
               weight_scale=1e-3,
               reg=0.0,
               dtype=np.float32):
    """
    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.params = {}
    self.reg = reg
    self.dtype = dtype


    ############################################################################
    # TODO: Initialize weights and biases for the three-layer convolutional    #
    # network. Weights should be initialized from a Gaussian with standard     #
    # deviation equal to weight_scale; biases should be initialized to zero.   #
    # All weights and biases should be stored in the dictionary self.params.   #
    # Store weights and biases for the convolutional layer using the keys 'W1' #
    # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
    # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
    # of the output affine layer.                                              #
    ############################################################################

    dist_normal = lambda size: np.random.normal(loc=0, scale=weight_scale, size=size)
    dist_zero = lambda size: np.zeros(size)

    C, H, W = input_dim
    num_conv_relu_pools = len(filters)
    num_affine_relus = len(hidden_dims)
    HH = WW = filter_size

    self.dim_cache = (C, H, W, HH, WW, filters, hidden_dims, num_classes)

    # [conv-relu-pool(2,2)]xN - [affine-relu]xM - affine - [softmax]
    for i in range(num_conv_relu_pools):
      F = filters[i]
      self.set_param('conv_weight', i, dist_normal((F, C, HH, WW)))
      self.set_param('conv_bias', i, dist_zero(F))
      C = F

    Hout = int(H / 2**num_conv_relu_pools)
    Wout = int(W / 2**num_conv_relu_pools)
    hd_input = F * Hout * Wout
    for i in range(num_affine_relus):
      hd = hidden_dims[i]
      self.set_param('affine_weight', i, dist_normal((hd_input, hd)))
      self.set_param('affine_bias', i, dist_zero(hd))
      hd_input = hd

    self.params['softmax_weight'] = dist_normal((hd_input, num_classes))
    self.params['softmax_bias'] = dist_zero(num_classes)

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    for k, v in self.params.items():
      print('init...', k, v.shape)
      self.params[k] = v.astype(dtype)


  def get_param(self, kind, num):
    return self.params[kind+str(num)]
  def set_param(self, kind, num, val):
    self.params[kind+str(num)] = val




  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.
    """
    C, H, W, HH, WW, filters, hidden_dims, num_classes = self.dim_cache
    N, C, H, W = X.shape

    # W1, b1 = self.params['W1'], self.params['b1']
    # W2, b2 = self.params['W2'], self.params['b2']
    # W3, b3 = self.params['W3'], self.params['b3']
    conv_param = {'stride': 1, 'pad': int((HH - 1) / 2)}
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}
    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the three-layer convolutional net,  #
    # computing the class scores for X and storing them in the scores          #
    # variable.                                                                #
    ############################################################################
    # [conv-relu-pool(2,2)]xN - [affine-relu]xM - affine - [softmax]


    crp_caches = []
    affine_caches = []
    num_conv_relu_pools = len(filters)
    num_affine_relus = len(hidden_dims)

    out = X
    for i in range(num_conv_relu_pools):  # conv layers
      W = self.get_param('conv_weight', i)
      b = self.get_param('conv_bias', i)

      out, crp_cache = conv_relu_pool_forward(out, W, b, conv_param, pool_param)
      crp_caches.append(crp_cache)
      N, F, Hout, Wout = out.shape

    out = out.reshape((N, -1))

    for i in range(num_affine_relus):  # affine layers
      W = self.get_param('affine_weight', i)
      b = self.get_param('affine_bias', i)

      out, affine_cache = affine_relu_forward(out, W, b)
      affine_caches.append(affine_cache)

    W = self.params['softmax_weight']
    b = self.params['softmax_bias']
    scores, scores_cache = affine_forward(out, W, b)

    if y is None:
      return scores

    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the three-layer convolutional net, #
    # storing the loss and gradients in the loss and grads variables. Compute  #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    ############################################################################

    scores -= np.max(scores, axis=1).reshape(N, 1)
    Sum_e_scores = np.sum(np.exp(scores), axis=1).reshape(N, 1)
    Probs = np.exp(scores) / Sum_e_scores
    loss = np.sum(-np.log(Probs[np.arange(N), y]))

    loss /= N
    W = self.params['softmax_weight']
    loss += 0.5 * self.reg * np.sum(W * W)
    for i in range(num_affine_relus):
      W = self.get_param('affine_weight', i)
      loss += 0.5 * self.reg * np.sum(W * W)
    for i in range(num_conv_relu_pools):
      W = self.get_param('conv_weight', i)
      loss += 0.5 * self.reg * np.sum(W * W)

    # end of loss, start grad

    Probs[np.arange(N), y] -= 1
    Probs /= N    # part dL/df

    dout, dw, db = affine_backward(Probs, scores_cache)
    grads['softmax_weight'] = dw + self.reg * self.params['softmax_weight']
    grads['softmax_bias'] = db

    zip_affine_caches = zip(range(num_affine_relus), affine_caches)
    for i, affine_cache in reversed(list(zip_affine_caches)):
      dout, dw, db = affine_relu_backward(dout, affine_cache)
      W = self.get_param('affine_weight', i)
      grads['affine_weight' + str(i)] = dw + self.reg * W
      grads['affine_bias' + str(i)] = db

    dout = dout.reshape((N, F, Hout, Wout))  # crp_out = crp_out.reshape((N, -1))
    zip_crp_caches = zip(range(num_conv_relu_pools), crp_caches)
    for i, crp_cache in reversed(list(zip_crp_caches)):
      dout, dw, db = conv_relu_pool_backward(dout, crp_cache)
      W = self.get_param('conv_weight', i)
      grads['conv_weight' + str(i)] = dw + self.reg * W
      grads['conv_bias' + str(i)] = db

    return loss, grads
