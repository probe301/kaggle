import numpy as np


def affine_forward(x, w, b):
  """
  Computes the forward pass for an affine (fully-connected) layer.

  The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
  examples, where each example x[i] has shape (d_1, ..., d_k). We will
  reshape each input into a vector of dimension D = d_1 * ... * d_k, and
  then transform it to an output vector of dimension M.

  Inputs:
  - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
  - w: A numpy array of weights, of shape (D, M)
  - b: A numpy array of biases, of shape (M,)

  Returns a tuple of:
  - out: output, of shape (N, M)
  - cache: (x, w, b)
  """
  out = None
  #############################################################################
  # TODO: Implement the affine forward pass.
  # Store the result in out.
  # You will need to reshape the input into rows.
  #############################################################################
  N = x.shape[0]
  x2 = x.reshape(N, -1)
  out = x2.dot(w) + b
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (x, w, b)
  return out, cache


def affine_backward(dout, cache):
  """
  Computes the backward pass for an affine layer.

  Inputs:
  - dout: Upstream derivative, of shape (N, M)
  - cache: Tuple of:
    - x: Input data, of shape (N, d_1, ... d_k)
    - w: Weights, of shape (D, M)

  Returns a tuple of:
  - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
  - dw: Gradient with respect to w, of shape (D, M)
  - db: Gradient with respect to b, of shape (M,)
  """
  x, w, b = cache
  dx, dw, db = None, None, None
  #############################################################################
  # TODO: Implement the affine backward pass.                                 #
  #############################################################################
  db = dout.sum(axis=0)
  N = x.shape[0]
  dw = x.reshape(N, -1).T.dot(dout)
  dx = dout.dot(w.T).reshape(*x.shape)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx, dw, db















def relu_forward(x):
  """
  Computes the forward pass for a layer of rectified linear units (ReLUs).

  Input:
  - x: Inputs, of any shape

  Returns a tuple of:
  - out: Output, of the same shape as x
  - cache: x
  """
  out = None
  #############################################################################
  # TODO: Implement the ReLU forward pass.                                    #
  #############################################################################
  out = x.clip(min=0)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = x
  return out, cache


def relu_backward(dout, cache):
  """
  Computes the backward pass for a layer of rectified linear units (ReLUs).

  Input:
  - dout: Upstream derivatives, of any shape
  - cache: Input x, of same shape as dout

  Returns:
  - dx: Gradient with respect to x
  """
  dx, x = None, cache
  #############################################################################
  # TODO: Implement the ReLU backward pass.                                   #
  #############################################################################
  mask = x > 0
  dx = dout * mask
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx









def batchnorm_forward(x, gamma, beta, bn_param):
  """
  Forward pass for batch normalization.

  During training the sample mean and (uncorrected) sample variance are
  computed from minibatch statistics and used to normalize the incoming data.
  During training we also keep an exponentially decaying running mean of the mean
  and variance of each feature, and these averages are used to normalize data
  at test-time.

  At each timestep we update the running averages for mean and variance using
  an exponential decay based on the momentum parameter:

  running_mean = momentum * running_mean + (1 - momentum) * sample_mean
  running_var = momentum * running_var + (1 - momentum) * sample_var

  Note that the batch normalization paper suggests a different test-time
  behavior: they compute sample mean and variance for each feature using a
  large number of training images rather than using a running average. For
  this implementation we have chosen to use running averages instead since
  they do not require an additional estimation step; the torch7 implementation
  of batch normalization also uses running averages.

  Input:
  - x: Data of shape (N, D)
  - gamma: Scale parameter of shape (D,)
  - beta: Shift paremeter of shape (D,)
  - bn_param: Dictionary with the following keys:
    - mode: 'train' or 'test'; required
    - eps: Constant for numeric stability
    - momentum: Constant for running mean / variance.
    - running_mean: Array of shape (D,) giving running mean of features
    - running_var Array of shape (D,) giving running variance of features

  Returns a tuple of:
  - out: of shape (N, D)
  - cache: A tuple of values needed in the backward pass
  """
  mode = bn_param['mode']
  eps = bn_param.get('eps', 1e-5)
  momentum = bn_param.get('momentum', 0.9)

  N, D = x.shape
  running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
  running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

  out, cache = None, None
  if mode == 'train':
    #############################################################################
    # TODO: Implement the training-time forward pass for batch normalization.   #
    # Use minibatch statistics to compute the mean and variance, use these      #
    # statistics to normalize the incoming data, and scale and shift the        #
    # normalized data using gamma and beta.                                     #
    #                                                                           #
    # You should store the output in the variable out. Any intermediates that   #
    # you need for the backward pass should be stored in the cache variable.    #
    #                                                                           #
    # You should also use your computed sample mean and variance together with  #
    # the momentum variable to update the running mean and running variance,    #
    # storing your result in the running_mean and running_var variables.        #
    #############################################################################
    # for copy βγεμσ²x̄√⋅⁄
    # ∑x = sum(x)
    # μ = 1/m * ∑x
    # x_u = x-u
    # σ² = 1/m * x_u²
    # √σ² = √(σ²+ε)
    # ÷√σ² = 1/√(σ²+ε)
    # x̄ = x_u * ÷√σ²
    # out = γ * x̄ + β


    sumx = np.sum(x, axis=0)
    μ = 1 / N * sumx
    x_u = x - μ
    x_u2 = x_u ** 2
    σ2 = 1 / N * np.sum(x_u2, axis=0)
    σ2ε = σ2 + eps
    sqrt_σ2 = np.sqrt(σ2ε)
    invert_sqrt_σ2 = 1 / sqrt_σ2
    x̄ = x_u * invert_sqrt_σ2
    out = gamma * x̄ + beta

    # sample_mean = μ
    # sample_var = σ2
    running_mean = momentum * running_mean + (1 - momentum) * μ
    running_var = momentum * running_var + (1 - momentum) * σ2
    cache = (sumx, μ, x_u, x_u2, σ2, σ2ε, sqrt_σ2, invert_sqrt_σ2, x̄, gamma, beta)
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
  elif mode == 'test':
    #############################################################################
    # TODO: Implement the test-time forward pass for batch normalization. Use   #
    # the running mean and variance to normalize the incoming data, then scale  #
    # and shift the normalized data using gamma and beta. Store the result in   #
    # the out variable.                                                         #
    #############################################################################
    x_norm = (x - running_mean) / np.sqrt(running_var + eps)
    out = x_norm * gamma + beta
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
  else:
    raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

  # Store the updated running means back into bn_param
  bn_param['running_mean'] = running_mean
  bn_param['running_var'] = running_var

  return out, cache





def batchnorm_backward(dout, cache):
  """
  Backward pass for batch normalization.

  For this implementation, you should write out a computation graph for
  batch normalization on paper and propagate gradients backward through
  intermediate nodes.

  Inputs:
  - dout: Upstream derivatives, of shape (N, D)
  - cache: Variable of intermediates from batchnorm_forward.

  Returns a tuple of:
  - dx: Gradient with respect to inputs x, of shape (N, D)
  - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
  - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
  """
  dx, dgamma, dbeta = None, None, None
  #############################################################################
  # TODO: Implement the backward pass for batch normalization. Store the      #
  # results in the dx, dgamma, and dbeta variables.                           #
  #############################################################################
  sumx, μ, x_μ, x_u2, σ2, σ2ε, sqrt_σ2, invert_sqrt_σ2, x̄, gamma, beta = cache
  N, D = dout.shape
  vsum = lambda x: np.sum(x, axis=0)
  vexpand = lambda x: np.ones([N, D]) * x

  dbeta = vsum(dout)
  dgamma = vsum(x̄ * dout)
  dx̄ = gamma * dout                 # [N, D]
  # assert dx̄.shape, (N, D)
  dinvert_sqrt_σ2 = vsum(x_μ * dx̄)  # from x_μ * invert_sqrt_σ2 = x̄
  # assert dinvert_sqrt_σ2.shape, (D)
  dx_μ_part1 = invert_sqrt_σ2 * dx̄  # from x_μ * invert_sqrt_σ2 = x̄ (part1)
  # assert dx_μ_part1.shape, (N, D)
  dsqrt_σ2 = -1 / (sqrt_σ2 ** 2) * dinvert_sqrt_σ2 # [D, ]
  # assert dsqrt_σ2.shape, (D, )
  dσ2 = 0.5 * (σ2ε ** -0.5) * dsqrt_σ2   # [D, ]   also, dε = same as dσ2
  # assert dσ2.shape, (D, )
  dx_u2 = 1/N * vexpand(dσ2)
  # assert dx_u2.shape, (N, D)
  dx_μ_part2 = 2 * x_μ * dx_u2       # [N, D] from sum x_μ² (part2)
  # assert dx_μ_part2.shape, (N, D)
  dx_μ = dx_μ_part1 + dx_μ_part2
  # assert dx_μ.shape, (N, D)
  dμ = - vsum(dx_μ)
  dx_part1 = dx_μ               # from x - μ (part1)

  dx_part2 = 1/N * vexpand(dμ)  # from μ = 1/m * sum(x) (part1)2
  dx = dx_part1 + dx_part2

  # 求平均运算: y = mean(x, axis=0)
  # 其中 x.shape = [N, D], y.shape =［D, ]
  # 梯度应为: dx = 1/N * np.ones([N, D]) * dy



  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return dx, dgamma, dbeta




def batchnorm_backward_alt(dout, cache):
  """
  Alternative backward pass for batch normalization.

  For this implementation you should work out the derivatives for the batch
  normalizaton backward pass on paper and simplify as much as possible. You
  should be able to derive a simple expression for the backward pass.

  Note: This implementation should expect to receive the same cache variable
  as batchnorm_backward, but might not use all of the values in the cache.

  Inputs / outputs: Same as batchnorm_backward
  """
  dx, dgamma, dbeta = None, None, None
  #############################################################################
  # TODO: Implement the backward pass for batch normalization. Store the      #
  # results in the dx, dgamma, and dbeta variables.                           #
  #                                                                           #
  # After computing the gradient with respect to the centered inputs, you     #
  # should be able to compute gradients with respect to the inputs in a       #
  # single statement; our implementation fits on a single 80-character line.  #
  #############################################################################
  pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return dx, dgamma, dbeta
















def dropout_forward(x, dropout_param):
  """
  Performs the forward pass for (inverted) dropout.

  Inputs:
  - x: Input data, of any shape
  - dropout_param: A dictionary with the following keys:
    - p: Dropout parameter. We drop each neuron output with probability p.
    - mode: 'test' or 'train'. If the mode is train, then perform dropout;
      if the mode is test, then just return the input.
    - seed: Seed for the random number generator. Passing seed makes this
      function deterministic, which is needed for gradient checking but not in
      real networks.

  Outputs:
  - out: Array of the same shape as x.
  - cache: A tuple (dropout_param, mask). In training mode, mask is the dropout
    mask that was used to multiply the input; in test mode, mask is None.
  """
  p, mode = dropout_param['p'], dropout_param['mode']
  if 'seed' in dropout_param:
    np.random.seed(dropout_param['seed'])

  mask = None
  out = None

  if mode == 'train':
    ###########################################################################
    # TODO: Implement the training phase forward pass for inverted dropout.   #
    # Store the dropout mask in the mask variable.                            #
    ###########################################################################
    # H1 = np.maximum(0, np.dot(W1, X) + b1)
    mask = (np.random.rand(*x.shape) < p) / p # 随机失活遮罩. 注意/p!
    out = x * mask # drop!
    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################
  elif mode == 'test':
    ###########################################################################
    # TODO: Implement the test phase forward pass for inverted dropout.       #
    ###########################################################################
    out = x
    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################

  cache = (dropout_param, mask)
  out = out.astype(x.dtype, copy=False)

  return out, cache


def dropout_backward(dout, cache):
  """
  Perform the backward pass for (inverted) dropout.

  Inputs:
  - dout: Upstream derivatives, of any shape
  - cache: (dropout_param, mask) from dropout_forward.
  """
  dropout_param, mask = cache
  mode = dropout_param['mode']

  dx = None
  if mode == 'train':
    ###########################################################################
    # TODO: Implement the training phase backward pass for inverted dropout.  #
    ###########################################################################
    dx = dout * mask
    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################
  elif mode == 'test':
    dx = dout
  return dx
















def conv_forward_naive(x, w, b, conv_param):
  """
  A naive implementation of the forward pass for a convolutional layer.

  The input consists of N data points, each with C channels, height H and width
  W. We convolve each input with F different filters, where each filter spans
  all C channels and has height HH and width WW.

  Input:
  - x: Input data of shape (N, C, H, W)
  - w: Filter weights of shape (F, C, HH, WW)
  - b: Biases, of shape (F,)
  - conv_param: A dictionary with the following keys:
    - 'stride': The number of pixels between adjacent receptive fields in the
      horizontal and vertical directions.
    - 'pad': The number of pixels that will be used to zero-pad the input.

  Returns a tuple of:
  - out: Output data, of shape (N, F, H', W') where H' and W' are given by
    H' = 1 + (H + 2 * pad - HH) / stride
    W' = 1 + (W + 2 * pad - WW) / stride
  - cache: (x, w, b, conv_param)
  """
  out = None
  #############################################################################
  # TODO: Implement the convolutional forward pass.                           #
  # Hint: you can use the function np.pad for padding.                        #
  #############################################################################
  stride = conv_param['stride']
  pad = conv_param['pad']
  N, C, H, W = x.shape
  F, C2, HH, WW = w.shape
  assert C == C2
  assert (H - HH + 2 * pad) % stride == 0
  assert (W - WW + 2 * pad) % stride == 0

  Hout = int((H - HH + 2 * pad) / stride) + 1 # 输出数据体空间尺寸(W-F +2P)/S+1
  Wout = int((W - WW + 2 * pad) / stride) + 1
  Field_body_size = C * HH * WW
  # 如果输入是[227x227x3]，要与尺寸为11x11x3的滤波器以步长为4进行卷积，
  # 就取输入中的[11x11x3]数据块，然后将其拉伸为长度为11x11x3=363的列向量
  # 重复进行这一过程，因为步长为4，所以输出的宽高为(227-11)/4+1=55
  # 所以得到im2col操作的输出矩阵X_col的尺寸是[363x3025]
  # 其中每列是拉伸的感受野，共有55x55=3025个
  # 注意因为感受野之间有重叠，所以输入数据体中的数字在不同的列中可能有重复
  # 卷积层的权重也同样被拉伸成行
  # 举例，有96个尺寸为[11x11x3]的滤波器，就生成一个矩阵W_row，尺寸为[96x363]
  # 现在卷积的结果和进行一个大矩阵乘np.dot(W_row, X_col)是等价的了
  # 这个操作的输出是[96x3025]，给出了每个滤波器在每个位置的点积输出
  # 结果最后必须被重新变为合理的输出尺寸[55x55x96]
  def x_to_col(x):
    # padding first, and x means 'only one data record'
    # x.shape = C, H, W
    cols = Hout * Wout
    ret = np.zeros((Field_body_size, cols))  # ret.shape = rows, cols
    # h, w means Field left up corner index
    cnt = 0
    for h in range(0, H+pad*2-HH+1, stride):
      for w in range(0, W+pad*2-WW+1, stride):
        volumn = x[:, h:h+HH, w:w+WW]
        ret[:, cnt] = volumn.reshape(Field_body_size)
        cnt += 1
    return ret

  Wr = w.reshape(F, Field_body_size)
  # arr = np.pad(arr, ((1, ), (0, )), 'constant', constant_values=0)
  # 第二个参数是每个axis的pad数,
  # 对于其中每个pad, 都分为before和after, 只写一个则是before=after
  Xpad = np.pad(x, ((0,), (0,), (pad,), (pad,)), 'constant', constant_values=0)

  out = np.zeros((N, F, Hout, Wout))  # out.shape N, F, Hout, Wout
  Xc = np.zeros((N, Field_body_size, Hout*Wout))
  for i in range(N):
    # Wr.shape = F, Field_body_size
    Xc[i] = x_to_col(Xpad[i])  # Xc[i].shape = Field_body_size, cols
    vol_out = np.dot(Wr, Xc[i]) + b.reshape(F, 1)  # F, cols
    vol_out = vol_out.reshape(F, Hout, Wout)
    out[i] = vol_out

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (x, w, b, conv_param, Xc)
  return out, cache


def conv_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a convolutional layer.

  Inputs:
  - dout: Upstream derivatives.
  - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

  Returns a tuple of:
  - dx: Gradient with respect to x
  - dw: Gradient with respect to w
  - db: Gradient with respect to b
  """
  dx, dw, db = None, None, None
  #############################################################################
  # TODO: Implement the convolutional backward pass.                          #
  #############################################################################

  def to_volumn(dXci):
    dXci       # shape Field_body_size, Hout*Wout
    volumn = np.zeros((C, H+2*pad, W+2*pad))
    cnt = 0
    for h in range(0, H+pad*2-HH+1, stride):
      for w in range(0, W+pad*2-WW+1, stride):
        body = dXci[:, cnt].reshape(C, HH, WW)
        volumn[:, h:h+HH, w:w+WW] += body
        cnt += 1
    # remove pad
    return volumn[:, pad:H+pad, pad:W+pad] # shape C, H, W

  x, w, b, conv_param, Xc = cache
  N, C, H, W = x.shape
  F, C, HH, WW = w.shape
  N, F, Hout, Wout = dout.shape
  Field_body_size = C * HH * WW

  stride, pad = conv_param['stride'], conv_param['pad']

  Wr = w.reshape(F, Field_body_size)
  dWr = np.zeros_like(Wr)
  dXc = np.zeros_like(Xc)
  dx, dw, db = np.zeros_like(x), np.zeros_like(w), np.zeros_like(b)

  for i in range(N):
    dout_i = dout[i]                 # shape F, Hout, Wout
    dout_i_t = dout_i.reshape(F, -1) # shape F, Hout*Wout
    db += dout_i_t.sum(axis=1)       # shape F,
    dWrXci = dout_i_t                # shape F, Hout*Wout
    dWr += np.dot(dWrXci, Xc[i].T)   # shape F, Field_body_size
    dXc[i] = np.dot(Wr.T, dWrXci)    # shape Field_body_size, Hout*Wout
    dx[i] = to_volumn(dXc[i])        # shape C, H, W

  dw = dWr.reshape(F, C, HH, WW)
  # dx /= N
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx, dw, db





def max_pool_forward_naive(x, pool_param):
  """
  A naive implementation of the forward pass for a max pooling layer.

  Inputs:
  - x: Input data, of shape (N, C, H, W)
  - pool_param: dictionary with the following keys:
    - 'pool_height': The height of each pooling region
    - 'pool_width': The width of each pooling region
    - 'stride': The distance between adjacent pooling regions

  Returns a tuple of:
  - out: Output data
  - cache: (x, pool_param)
  """
  out = None
  #############################################################################
  # TODO: Implement the max pooling forward pass                              #
  #############################################################################
  poolh = pool_param['pool_height']
  poolw = pool_param['pool_width']
  stride = pool_param['stride']
  N, C, H, W = x.shape

  def pool(x_piece):
    # x_piece.shape = H, W
    poolx = np.zeros((Hout*Wout, ))
    pos_poolx = np.zeros((Hout*Wout, ))
    cnt = 0
    for h in range(0, H-poolh+1, stride):
      for w in range(0, W-poolw+1, stride):
        block = x_piece[h:h+poolh, w:w+poolw]
        poolx[cnt] = np.amax(block)  # max of value
        pos_poolx[cnt] = np.argmax(block) # max of indice
        cnt += 1
    return poolx, pos_poolx

  assert (H - poolh) % stride == 0
  assert (W - poolw) % stride == 0
  Hout = int((H - poolh) / stride + 1)
  Wout = int((W - poolw) / stride + 1)

  out = np.zeros((N, C, Hout*Wout))
  pos = np.zeros((N, C, Hout*Wout))
  for n in range(N):
    for c in range(C):
      poolx, pos_poolx = pool(x[n, c])
      out[n, c, :] = poolx
      pos[n, c, :] = pos_poolx
  out = out.reshape(N, C, Hout, Wout)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (x, pool_param, pos)
  return out, cache

def max_pool_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a max pooling layer.

  Inputs:
  - dout: Upstream derivatives
  - cache: A tuple of (x, pool_param) as in the forward pass.

  Returns:
  - dx: Gradient with respect to x
  """
  dx = None
  #############################################################################
  # TODO: Implement the max pooling backward pass                             #
  #############################################################################
  x, pool_param, pos = cache
  poolh = pool_param['pool_height']
  poolw = pool_param['pool_width']
  stride = pool_param['stride']
  N, C, H, W = x.shape
  N, C, Hout, Wout = dout.shape

  def unpool(dout_piece, pos_piece):
    # dout_piece.shape Hout, Wout
    # pos_piece.shape Hout*Wout
    dout_piece = dout_piece.reshape(Hout*Wout)
    dx_piece = np.zeros((H, W))
    cnt = 0
    for h in range(0, H-poolh+1, stride):
      for w in range(0, W-poolw+1, stride):
        vmax = dout_piece[cnt]
        imax = int(pos_piece[cnt])
        dblock = np.zeros((poolh*poolw))
        dblock[imax] = vmax
        dx_piece[h:h+poolh, w:w+poolw] = dblock.reshape(poolh, poolw)
        cnt += 1
    return dx_piece

  dx = np.zeros_like(x)
  for n in range(N):
    for c in range(C):
      dout_piece = dout[n, c]
      pos_piece = pos[n, c]
      dx[n, c] = unpool(dout_piece, pos_piece)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx



def print_shape(x):
  import sys
  local = sys._getframe(1).f_locals
  print(x+'.shape=', local.get(x).shape)












def spatial_batchnorm_forward(x, gamma, beta, bn_param):
  """
  Computes the forward pass for spatial batch normalization.

  Inputs:
  - x: Input data of shape (N, C, H, W)
  - gamma: Scale parameter, of shape (C,)
  - beta: Shift parameter, of shape (C,)
  - bn_param: Dictionary with the following keys:
    - mode: 'train' or 'test'; required
    - eps: Constant for numeric stability
    - momentum: Constant for running mean / variance. momentum=0 means that
      old information is discarded completely at every time step, while
      momentum=1 means that new information is never incorporated. The
      default of momentum=0.9 should work well in most situations.
    - running_mean: Array of shape (D,) giving running mean of features
    - running_var Array of shape (D,) giving running variance of features

  Returns a tuple of:
  - out: Output data, of shape (N, C, H, W)
  - cache: Values needed for the backward pass
  """
  out, cache = None, None

  #############################################################################
  # TODO: Implement the forward pass for spatial batch normalization.         #
  #                                                                           #
  # HINT: You can implement spatial batch normalization using the vanilla     #
  # version of batch normalization defined above. Your implementation should  #
  # be very short; ours is less than five lines.                              #
  #############################################################################
  pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return out, cache


def spatial_batchnorm_backward(dout, cache):
  """
  Computes the backward pass for spatial batch normalization.

  Inputs:
  - dout: Upstream derivatives, of shape (N, C, H, W)
  - cache: Values from the forward pass

  Returns a tuple of:
  - dx: Gradient with respect to inputs, of shape (N, C, H, W)
  - dgamma: Gradient with respect to scale parameter, of shape (C,)
  - dbeta: Gradient with respect to shift parameter, of shape (C,)
  """
  dx, dgamma, dbeta = None, None, None

  #############################################################################
  # TODO: Implement the backward pass for spatial batch normalization.        #
  #                                                                           #
  # HINT: You can implement spatial batch normalization using the vanilla     #
  # version of batch normalization defined above. Your implementation should  #
  # be very short; ours is less than five lines.                              #
  #############################################################################
  pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return dx, dgamma, dbeta


def svm_loss(x, y):
  """
  Computes the loss and gradient using for multiclass SVM classification.

  Inputs:
  - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
    for the ith input.
  - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
    0 <= y[i] < C

  Returns a tuple of:
  - loss: Scalar giving the loss
  - dx: Gradient of the loss with respect to x
  """
  N = x.shape[0]
  correct_class_scores = x[np.arange(N), y]
  margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
  margins[np.arange(N), y] = 0
  loss = np.sum(margins) / N
  num_pos = np.sum(margins > 0, axis=1)
  dx = np.zeros_like(x)
  dx[margins > 0] = 1
  dx[np.arange(N), y] -= num_pos
  dx /= N
  return loss, dx


def softmax_loss(x, y):
  """
  Computes the loss and gradient for softmax classification.

  Inputs:
  - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
    for the ith input.
  - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
    0 <= y[i] < C

  Returns a tuple of:
  - loss: Scalar giving the loss
  - dx: Gradient of the loss with respect to x
  """
  probs = np.exp(x - np.max(x, axis=1, keepdims=True))
  probs /= np.sum(probs, axis=1, keepdims=True)
  N = x.shape[0]
  loss = -np.sum(np.log(probs[np.arange(N), y])) / N
  dx = probs.copy()
  dx[np.arange(N), y] -= 1
  dx /= N
  return loss, dx
