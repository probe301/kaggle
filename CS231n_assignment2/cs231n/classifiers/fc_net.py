import numpy as np

from cs231n.layers import *
from cs231n.layer_utils import *


class TwoLayerNet(object):
  """
  A two-layer fully-connected neural network with ReLU nonlinearity and
  softmax loss that uses a modular layer design. We assume an input dimension
  of D, a hidden dimension of H, and perform classification over C classes.

  The architecure should be affine - relu - affine - softmax.

  Note that this class does not implement gradient descent; instead, it
  will interact with a separate Solver object that is responsible for running
  optimization.

  The learnable parameters of the model are stored in the dictionary
  self.params that maps parameter names to numpy arrays.
  """

  def __init__(self,
               input_dim=3*32*32,
               hidden_dim=100,
               num_classes=10,
               weight_scale=1e-3, reg=0.0):
    """
    Initialize a new network.

    Inputs:
    - input_dim: An integer giving the size of the input
    - hidden_dim: An integer giving the size of the hidden layer
    - num_classes: An integer giving the number of classes to classify
    - dropout: Scalar between 0 and 1 giving dropout strength.
    - weight_scale: Scalar giving the standard deviation for random
      initialization of the weights.
    - reg: Scalar giving L2 regularization strength.
    """
    self.params = {}
    self.reg = reg

    ############################################################################
    # TODO: Initialize the weights and biases of the two-layer net. Weights    #
    # should be initialized from a Gaussian with standard deviation equal to   #
    # weight_scale, and biases should be initialized to zero. All weights and  #
    # biases should be stored in the dictionary self.params, with first layer  #
    # weights and biases using the keys 'W1' and 'b1' and second layer weights #
    # and biases using the keys 'W2' and 'b2'.                                 #
    ############################################################################
    self.params['W1'] = np.random.normal(loc=0, scale=weight_scale,
                                         size=(input_dim, hidden_dim))
    self.params['b1'] = np.zeros(hidden_dim)
    self.params['W2'] = np.random.normal(loc=0, scale=weight_scale,
                                         size=(hidden_dim, num_classes))
    self.params['b2'] = np.zeros(num_classes)
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################


  def loss(self, X, y=None):
    """
    Compute loss and gradient for a minibatch of data.

    Inputs:
    - X: Array of input data of shape (N, d_1, ..., d_k)
    - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

    Returns:
    If y is None, then run a test-time forward pass of the model and return:
    - scores: Array of shape (N, C) giving classification scores, where
      scores[i, c] is the classification score for X[i] and class c.

    If y is not None, then run a training-time forward and backward pass and
    return a tuple of:
    - loss: Scalar value giving the loss
    - grads: Dictionary with the same keys as self.params, mapping parameter
      names to gradients of the loss with respect to those parameters.
    """
    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the two-layer net, computing the    #
    # class scores for X and storing them in the scores variable.              #
    ############################################################################
    pass
    # out, cache = affine_relu_forward(x, w, b)
    # dx, dw, db = affine_relu_backward(dout, cache)
    N = X.shape[0]
    X = X.reshape(N, -1)
    W1 = self.params['W1']
    b1 = self.params['b1']
    W2 = self.params['W2']
    b2 = self.params['b2']
    H1_out, H1_cache = affine_relu_forward(X, W1, b1)
    scores, scores_cache = affine_forward(H1_out, W2, b2)
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    # If y is None then we are in test mode so just return scores
    if y is None:
      return scores

    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the two-layer net. Store the loss  #
    # in the loss variable and gradients in the grads dictionary. Compute data #
    # loss using softmax, and make sure that grads[k] holds the gradients for  #
    # self.params[k]. Don't forget to add L2 regularization!                   #
    #                                                                          #
    # NOTE: To ensure that your implementation matches ours and you pass the   #
    # automated tests, make sure that your L2 regularization includes a factor #
    # of 0.5 to simplify the expression for the gradient.                      #
    ############################################################################
    scores -= np.max(scores, axis=1).reshape(N, 1)
    Sum_e_scores = np.sum(np.exp(scores), axis=1).reshape(N, 1)
    Probs = np.exp(scores) / Sum_e_scores
    loss = np.sum(-np.log(Probs[np.arange(N), y]))

    loss /= N
    loss += 0.5 * self.reg * np.sum(W1 * W1)
    loss += 0.5 * self.reg * np.sum(W2 * W2)



    Probs[np.arange(N), y] -= 1
    Probs /= N    # part dL/df
    grad_b2 = np.sum(Probs, axis=0)
    grad_W2 = H1_out.T.dot(Probs) + self.reg * W2
    grad_M = Probs.dot(W2.T)     # M: Max(X * W1 + b1)
    H1_mask = H1_out > 0
    grad_b1 = np.sum(grad_M * H1_mask, axis=0)
    grad_W1 = X.T.dot(grad_M * H1_mask) + self.reg * W1
    grads['W1'] = grad_W1
    grads['b1'] = grad_b1
    grads['W2'] = grad_W2
    grads['b2'] = grad_b2

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return loss, grads














def affine_batchnorm_relu_forward(x, w, b, gamma, beta, bn_param):
  """
  Convenience layer that perorms an affine transform followed by a ReLU
  """
  out, fc_cache = affine_forward(x, w, b)
  out, bn_cache = batchnorm_forward(out, gamma, beta, bn_param)
  out, relu_cache = relu_forward(out)
  cache = (fc_cache, bn_cache, relu_cache)
  return out, cache

def affine_batchnorm_relu_backward(dout, cache):
  """
  Backward pass for the affine-relu convenience layer
  """
  fc_cache, bn_cache, relu_cache = cache
  dout = relu_backward(dout, relu_cache)
  dout, dgamma, dbeta = batchnorm_backward(dout, bn_cache)
  dout, dw, db = affine_backward(dout, fc_cache)
  return dout, dw, db, dgamma, dbeta




def affine_relu_dropout_forward(x, w, b, dropout_param):
  out, fc_cache = affine_forward(x, w, b)
  out, relu_cache = relu_forward(out)
  out, dropout_cache = dropout_forward(out, dropout_param)
  cache = (fc_cache, relu_cache, dropout_cache)
  return out, cache

def affine_relu_dropout_backward(dout, cache):
  fc_cache, relu_cache, dropout_cache = cache
  dout = dropout_backward(dout, dropout_cache)
  dout = relu_backward(dout, relu_cache)
  dout, dw, db = affine_backward(dout, fc_cache)
  return dout, dw, db








class FullyConnectedNet(object):
  """
  A fully-connected neural network with an arbitrary number of hidden layers,
  ReLU nonlinearities, and a softmax loss function. This will also implement
  dropout and batch normalization as options. For a network with L layers,
  the architecture will be

  {affine - [batch norm] - relu - [dropout]} x (L - 1) - affine - softmax

  where batch normalization and dropout are optional, and the {...} block is
  repeated L - 1 times.

  Similar to the TwoLayerNet above, learnable parameters are stored in the
  self.params dictionary and will be learned using the Solver class.
  """

  def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
               dropout=0, use_batchnorm=False, reg=0.0,
               weight_scale=1e-2, dtype=np.float32, seed=None):
    """
    Initialize a new FullyConnectedNet.

    Inputs:
    - hidden_dims: A list of integers giving the size of each hidden layer.
    - input_dim: An integer giving the size of the input.
    - num_classes: An integer giving the number of classes to classify.
    - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=0 then
      the network should not use dropout at all.
    - use_batchnorm: Whether or not the network should use batch normalization.
    - reg: Scalar giving L2 regularization strength.
    - weight_scale: Scalar giving the standard deviation for random
      initialization of the weights.
    - dtype: A numpy datatype object; all computations will be performed using
      this datatype. float32 is faster but less accurate, so you should use
      float64 for numeric gradient checking.
    - seed: If not None, then pass this random seed to the dropout layers. This
      will make the dropout layers deteriminstic so we can gradient check the
      model.
    """
    self.use_batchnorm = use_batchnorm
    self.use_dropout = dropout > 0
    assert not (self.use_dropout and self.use_batchnorm)
    self.reg = reg
    self.num_layers = 1 + len(hidden_dims)
    self.dtype = dtype
    self.params = {}
    self.weight_scale = weight_scale

    ############################################################################
    # TODO: Initialize the parameters of the network, storing all values in    #
    # the self.params dictionary. Store weights and biases for the first layer #
    # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
    # initialized from a normal distribution with standard deviation equal to  #
    # weight_scale and biases should be initialized to zero.                   #
    #                                                                          #
    # When using batch normalization, store scale and shift parameters for the #
    # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
    # beta2, etc. Scale parameters should be initialized to one and shift      #
    # parameters should be initialized to zero.                                #
    ############################################################################
    dist_normal = lambda a, b: np.random.normal(loc=0, scale=weight_scale, size=(a, b))

    # 建议将神经元的权重向量初始化为：
    # w = np.random.randn(n) / sqrt(n)。
    # 其中n是输入数据的数量。
    # 这样就保证了网络中所有神经元起始时有近似同样的输出分布。
    # 实践经验证明，这样做可以提高收敛的速度。
    # w = np.random.randn(n) * sqrt(2.0/n)。
    # 这个形式是神经网络算法使用ReLU神经元时的当前最佳推荐
    dist_zero = lambda size: np.zeros(size)
    dist_ones = lambda size: np.ones(size)

    first_dim = input_dim
    for i, hidden_dim in enumerate(hidden_dims, 1):
      # print('init hidden ', i, first_dim, hidden_dim)
      self.params['W' + str(i)] = dist_normal(first_dim, hidden_dim)
      self.params['b' + str(i)] = dist_zero(hidden_dim)
      if self.use_batchnorm:
        self.params['gamma' + str(i)] = dist_ones(hidden_dim)  # for batch norm
        self.params['beta' + str(i)] = dist_zero(hidden_dim)

      first_dim = hidden_dim
    self.params['W' + str(i+1)] = dist_normal(first_dim, num_classes)
    self.params['b' + str(i+1)] = dist_zero(num_classes)

    # print('init parmas W1', self.params['W1'])
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    # When using dropout we need to pass a dropout_param dictionary to each
    # dropout layer so that the layer knows the dropout probability and the mode
    # (train / test). You can pass the same dropout_param to each dropout layer.
    self.dropout_param = {}
    if self.use_dropout:
      self.dropout_param = {'mode': 'train', 'p': dropout}
      if seed is not None:
        self.dropout_param['seed'] = seed

    # With batch normalization we need to keep track of running means and
    # variances, so we need to pass a special bn_param object to each batch
    # normalization layer. You should pass self.bn_params[0] to the forward pass
    # of the first batch normalization layer, self.bn_params[1] to the forward
    # pass of the second batch normalization layer, etc.
    self.bn_params = []
    if self.use_batchnorm:
      self.bn_params = [{'mode': 'train'} for i in range(self.num_layers - 1)]

    # Cast all parameters to the correct datatype
    for k, v in self.params.items():
      self.params[k] = v.astype(dtype)


  def loss(self, X, y=None):
    """
    Compute loss and gradient for the fully-connected net.

    Input / output: Same as TwoLayerNet above.
    """
    X = X.astype(self.dtype)
    mode = 'test' if y is None else 'train'

    # Set train/test mode for batchnorm params and dropout param since they
    # behave differently during training and testing.
    if self.dropout_param is not None:
      self.dropout_param['mode'] = mode

    if self.use_batchnorm:
      for bn_param in self.bn_params:
        bn_param['mode'] = mode

    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the fully-connected net, computing  #
    # the class scores for X and storing them in the scores variable.          #
    #                                                                          #
    # When using dropout, you'll need to pass self.dropout_param to each       #
    # dropout forward pass.                                                    #
    #                                                                          #
    # When using batch normalization, you'll need to pass self.bn_params[0] to #
    # the forward pass for the first batch normalization layer, pass           #
    # self.bn_params[1] to the forward pass for the second batch normalization #
    # layer, etc.                                                              #
    ############################################################################
    Weight = lambda i: self.params['W' + str(i)]
    bias = lambda i: self.params['b' + str(i)]
    gamma = lambda i: self.params['gamma' + str(i)]
    beta = lambda i: self.params['beta' + str(i)]
    forward_caches = {}

    N = X.shape[0]
    X = X.reshape(N, -1)
    # W1 = self.params['W1']
    # b1 = self.params['b1']
    # W2 = self.params['W2']
    # b2 = self.params['b2']
    out = X
    for i in range(1, self.num_layers):
      if self.use_batchnorm:
        list_params = out, Weight(i), bias(i), gamma(i), beta(i), self.bn_params[i-1]
        out, hidden_cache = affine_batchnorm_relu_forward(*list_params)
        forward_caches[i] = hidden_cache
      elif self.use_dropout:
        out, hidden_cache = affine_relu_dropout_forward(out, Weight(i), bias(i), self.dropout_param)
        forward_caches[i] = hidden_cache
      else:
        out, hidden_cache = affine_relu_forward(out, Weight(i), bias(i))
        forward_caches[i] = hidden_cache

    # last layer
    scores, scores_cache = affine_forward(out, Weight(i+1), bias(i+1))
    forward_caches[i+1] = scores_cache

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    # If test mode return early
    if mode == 'test':
      return scores

    loss, grads = 0.0, {}
    ############################################################################
    # TODO: Implement the backward pass for the fully-connected net. Store the #
    # loss in the loss variable and gradients in the grads dictionary. Compute #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    #                                                                          #
    # When using batch normalization, you don't need to regularize the scale   #
    # and shift parameters.                                                    #
    #                                                                          #
    # NOTE: To ensure that your implementation matches ours and you pass the   #
    # automated tests, make sure that your L2 regularization includes a factor #
    # of 0.5 to simplify the expression for the gradient.                      #
    ############################################################################
    scores -= np.max(scores, axis=1).reshape(N, 1)
    Sum_e_scores = np.sum(np.exp(scores), axis=1).reshape(N, 1)
    Probs = np.exp(scores) / Sum_e_scores
    loss = np.sum(-np.log(Probs[np.arange(N), y]))

    loss /= N
    for i in range(1, self.num_layers):
      loss += 0.5 * self.reg * np.sum(Weight(i) * Weight(i))
    loss += 0.5 * self.reg * np.sum(Weight(i+1) * Weight(i+1))



    Probs[np.arange(N), y] -= 1
    Probs /= N    # part dL/df


    dout, dw, db = affine_backward(Probs, forward_caches[i+1])
    grads['W' + str(i+1)] = dw + self.reg * Weight(i+1)
    grads['b' + str(i+1)] = db

    for i in reversed(range(1, self.num_layers)):

      if self.use_batchnorm:
        dout, dw, db, dgamma, dbeta = affine_batchnorm_relu_backward(dout, forward_caches[i])
        grads['W' + str(i)] = dw + self.reg * Weight(i)
        grads['b' + str(i)] = db
        grads['gamma' + str(i)] = dgamma
        grads['beta' + str(i)] = dbeta
      elif self.use_dropout:
        dout, dw, db = affine_relu_dropout_backward(dout, forward_caches[i])
        grads['W' + str(i)] = dw + self.reg * Weight(i)
        grads['b' + str(i)] = db
      else:
        dout, dw, db = affine_relu_backward(dout, forward_caches[i])
        grads['W' + str(i)] = dw + self.reg * Weight(i)
        grads['b' + str(i)] = db

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return loss, grads
