import numpy as np


"""
This file defines layer types that are commonly used for recurrent neural
networks.
"""


def print_shape(x):
  import sys
  local = sys._getframe(1).f_locals
  print(x+'.shape=', local.get(x).shape)


def rnn_step_forward(x, prev_h, Wx, Wh, b):
  """
  Run the forward pass for a single timestep of a vanilla RNN that uses a tanh
  activation function.

  The input data has dimension D, the hidden state has dimension H, and we use
  a minibatch size of N.

  Inputs:
  - x: Input data for this timestep, of shape (N, D).
  - prev_h: Hidden state from previous timestep, of shape (N, H)
  - Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
  - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
  - b: Biases of shape (H,)

  Returns a tuple of:
  - next_h: Next hidden state, of shape (N, H)
  - cache: Tuple of values needed for the backward pass.
  """
  next_h, cache = None, None
  ##############################################################################
  # TODO: Implement a single forward step for the vanilla RNN. Store the next  #
  # hidden state and any values you need for the backward pass in the next_h   #
  # and cache variables respectively.                                          #
  ##############################################################################
  x_dot_Wx = x.dot(Wx)
  prev_h_dot_Wh = prev_h.dot(Wh)
  xWx_prevhWh_b = x_dot_Wx + prev_h_dot_Wh + b
  next_h = np.tanh(xWx_prevhWh_b)
  cache = (x, prev_h, Wx, Wh, xWx_prevhWh_b)
  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################
  return next_h, cache


def rnn_step_backward(dnext_h, cache):
  """
  Backward pass for a single timestep of a vanilla RNN.

  Inputs:
  - dnext_h: Gradient of loss with respect to next hidden state
  - cache: Cache object from the forward pass

  Returns a tuple of:
  - dx: Gradients of input data, of shape (N, D)
  - dprev_h: Gradients of previous hidden state, of shape (N, H)
  - dWx: Gradients of input-to-hidden weights, of shape (N, H)
  - dWh: Gradients of hidden-to-hidden weights, of shape (H, H)
  - db: Gradients of bias vector, of shape (H,)
  """
  dx, dprev_h, dWx, dWh, db = None, None, None, None, None
  ##############################################################################
  # TODO: Implement the backward pass for a single step of a vanilla RNN.      #
  #                                                                            #
  # HINT: For the tanh function, you can compute the local derivative in terms #
  # of the output value from tanh.                                             #
  ##############################################################################
  (x, prev_h, Wx, Wh, xWx_prevhWh_b) = cache
  dxWx_prevhWh_b = (1 - np.tanh(xWx_prevhWh_b) ** 2) * dnext_h  # shape N, H
  db = np.sum(dxWx_prevhWh_b, axis=0)
  dx = dxWx_prevhWh_b.dot(Wx.T)
  dWx = x.T.dot(dxWx_prevhWh_b)
  dprev_h = dxWx_prevhWh_b.dot(Wh.T)
  dWh = prev_h.T.dot(dxWx_prevhWh_b)
  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################
  return dx, dprev_h, dWx, dWh, db


def rnn_forward(x, h0, Wx, Wh, b):
  """
  Run a vanilla RNN forward on an entire sequence of data. We assume an input
  sequence composed of T vectors, each of dimension D. The RNN uses a hidden
  size of H, and we work over a minibatch containing N sequences. After running
  the RNN forward, we return the hidden states for all timesteps.
  Inputs:
  - x: Input data for the entire timeseries, of shape (N, T, D).
  - h0: Initial hidden state, of shape (N, H)
  - Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
  - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
  - b: Biases of shape (H,)
  Returns a tuple of:
  - h: Hidden states for the entire timeseries, of shape (N, T, H).
  - cache: Values needed in the backward pass
  """
  h, cache = None, None
  ##############################################################################
  # TODO: Implement forward pass for a vanilla RNN running on a sequence of    #
  # input data. You should use the rnn_step_forward function that you defined  #
  # above.                                                                     #
  ##############################################################################
  N, T, D = x.shape
  N, H = h0.shape
  h = np.zeros((N, T, H))
  next_h = h0
  cache_dict = dict()
  for t in range(T):
    next_h, cache_step = rnn_step_forward(x[:, t, :], next_h, Wx, Wh, b)
    h[:, t, :] = next_h
    cache_dict[t] = cache_step  # include (step) x, prev_h, Wx, Wh, xWx_prevhWh_b
  cache = x, h0, Wx, Wh, b, cache_dict
  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################
  return h, cache


def rnn_backward(dh, cache):
  """
  Compute the backward pass for a vanilla RNN over an entire sequence of data.

  Inputs:
  - dh: Upstream gradients of all hidden states, of shape (N, T, H)

  Returns a tuple of:
  - dx: Gradient of inputs, of shape (N, T, D)
  - dh0: Gradient of initial hidden state, of shape (N, H)
  - dWx: Gradient of input-to-hidden weights, of shape (D, H)
  - dWh: Gradient of hidden-to-hidden weights, of shape (H, H)
  - db: Gradient of biases, of shape (H,)
  """
  dx, dh0, dWx, dWh, db = None, None, None, None, None
  ##############################################################################
  # TODO: Implement the backward pass for a vanilla RNN running an entire      #
  # sequence of data. You should use the rnn_step_backward function that you   #
  # defined above.                                                             #
  ##############################################################################
  N, T, H = dh.shape
  x, h0, Wx, Wh, b, cache_dict = cache
  N, T, D = x.shape

  dx = np.zeros_like(x)
  dWx = np.zeros_like(Wx)
  dWh = np.zeros_like(Wh)
  db = np.zeros_like(b)

  dprev_h = np.zeros_like(h0)

  for t in reversed(range(T)):
    dxstp, dprev_h, dWxstp, dWhstp, dbstp = \
        rnn_step_backward(dh[:, t, :] + dprev_h, cache_dict[t])
    # 每个时间 t 对应的 dh 实际上有两个来源
    # 第一是该 h 输出给 y 的(比如输出为一个字母的 error)
    # 第二是该 h 输出到 t+1 时刻的 h
    # 本函数的参数 "dh" 实际上就是第一类
    # 因此在计算反传时, 还得累计第二类, 即 dprev_h (从下一时刻 h 传回来的 dh)
    # 这就是为什么用 rnn_step_backward(dh[:, t, :] + dprev_h, ...)
    dx[:, t, :] = dxstp
    dWx += dWxstp
    dWh += dWhstp
    db += dbstp

  dh0 = dprev_h
  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################
  return dx, dh0, dWx, dWh, db
















def word_embedding_forward(x, W):
  """
  Forward pass for word embeddings. We operate on minibatches of size N where
  each sequence has length T. We assume a vocabulary of V words, assigning each
  to a vector of dimension D.

  Inputs:
  - x: Integer array of shape (N, T) giving indices of words. Each element idx
    of x muxt be in the range 0 <= idx < V.
  - W: Weight matrix of shape (V, D) giving word vectors for all words.

  Returns a tuple of:
  - out: Array of shape (N, T, D) giving word vectors for all input words.
  - cache: Values needed for the backward pass
  """
  out, cache = None, None
  ##############################################################################
  # TODO: Implement the forward pass for word embeddings.                      #
  #                                                                            #
  # HINT: This should be very simple.                                          #
  ##############################################################################
  N, T = x.shape
  V, D = W.shape
  out = W[x.reshape(N * T)].reshape(N, T, D)
  cache = (x, W)
  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################
  return out, cache


def word_embedding_backward(dout, cache):
  """
  Backward pass for word embeddings. We cannot back-propagate into the words
  since they are integers, so we only return gradient for the word embedding
  matrix.

  HINT: Look up the function np.add.at

        from numpy doc
        Set items 0 and 1 to their negative values:
        >>> a = np.array([1, 2, 3, 4])
        >>> np.negative.at(a, [0, 1])
        >>> print(a)
        array([-1, -2, 3, 4])

        Increment items 0 and 1, and increment item 2 twice:
        >>> a = np.array([1, 2, 3, 4])
        >>> np.add.at(a, [0, 1, 2, 2], 1)
        >>> print(a)
        array([2, 3, 5, 4])

        Add items 0 and 1 in first array to second array,
        and store results in first array:
        >>> a = np.array([1, 2, 3, 4])
        >>> b = np.array([1, 2])
        >>> np.add.at(a, [0, 1], b)
        >>> print(a)
        array([2, 4, 3, 4])

  Inputs:
  - dout: Upstream gradients of shape (N, T, D)
  - cache: Values from the forward pass

  Returns:
  - dW: Gradient of word embedding matrix, of shape (V, D).
  """
  dW = None
  ##############################################################################
  # TODO: Implement the backward pass for word embeddings.                     #
  #                                                                            #
  # HINT: Look up the function np.add.at                                       #
  ##############################################################################
  (x, W) = cache
  N, T = x.shape
  V, D = W.shape
  dW = np.zeros_like(W)
  np.add.at(dW, x.reshape(N*T, ), dout.reshape(N*T, D))

  # for n in range(N):   # 更容易理解的做法
  #   for t in range(T):
  #     idx = x[n, t]
  #     dW[idx] += dout[n, t]
  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################
  return dW












def sigmoid(x):
  """
  A numerically stable version of the logistic sigmoid function.
  """
  pos_mask = (x >= 0)
  neg_mask = (x < 0)
  z = np.zeros_like(x)
  z[pos_mask] = np.exp(-x[pos_mask])
  z[neg_mask] = np.exp(x[neg_mask])
  top = np.ones_like(x)
  top[neg_mask] = z[neg_mask]
  return top / (1 + z)









def lstm_step_forward(x, prev_h, prev_c, Wx, Wh, b):
  """
  Forward pass for a single timestep of an LSTM.

  The input data has dimension D, the hidden state has dimension H, and we use
  a minibatch size of N.

  Inputs:
  - x: Input data, of shape (N, D)
  - prev_h: Previous hidden state, of shape (N, H)
  - prev_c: previous cell state, of shape (N, H)
  - Wx: Input-to-hidden weights, of shape (D, 4H)
  - Wh: Hidden-to-hidden weights, of shape (H, 4H)
  - b: Biases, of shape (4H,)

  Returns a tuple of:
  - next_h: Next hidden state, of shape (N, H)
  - next_c: Next cell state, of shape (N, H)
  - cache: Tuple of values needed for backward pass.
  """
  next_h, next_c, cache = None, None, None
  #############################################################################
  # TODO: Implement the forward pass for a single timestep of an LSTM.        #
  # You may want to use the numerically stable sigmoid implementation above.  #
  #############################################################################
  N, H = prev_h.shape
  a = x.dot(Wx) + prev_h.dot(Wh) + b
  sig_ainput = sigmoid(a[:, :H])
  sig_aforget = sigmoid(a[:, H:2*H])
  sig_aoutput = sigmoid(a[:, 2*H:3*H])
  tanh_block_input = np.tanh(a[:, 3*H:])

  next_c = sig_aforget * prev_c + sig_ainput * tanh_block_input
  tanh_next_c = np.tanh(next_c)
  next_h = sig_aoutput * tanh_next_c

  cache = (x, prev_h, prev_c, Wx, Wh, a, sig_ainput, sig_aforget, sig_aoutput, tanh_block_input, tanh_next_c)
  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################

  return next_h, next_c, cache


def lstm_step_backward(dnext_h, dnext_c, cache):
  """
  Backward pass for a single timestep of an LSTM.

  Inputs:
  - dnext_h: Gradients of next hidden state, of shape (N, H)
  - dnext_c: Gradients of next cell state, of shape (N, H)
  - cache: Values from the forward pass

  Returns a tuple of:
  - dx: Gradient of input data, of shape (N, D)
  - dprev_h: Gradient of previous hidden state, of shape (N, H)
  - dprev_c: Gradient of previous cell state, of shape (N, H)
  - dWx: Gradient of input-to-hidden weights, of shape (D, 4H)
  - dWh: Gradient of hidden-to-hidden weights, of shape (H, 4H)
  - db: Gradient of biases, of shape (4H,)
  """
  dx, dh, dc, dWx, dWh, db = None, None, None, None, None, None
  #############################################################################
  # TODO: Implement the backward pass for a single timestep of an LSTM.       #
  #                                                                           #
  # HINT: For sigmoid and tanh you can compute local derivatives in terms of  #
  # the output value from the nonlinearity.                                   #
  #############################################################################
  pass
  def dtanh(tanhx=None, x=None):
    if tanhx is None:
      tanhx = np.tanh(x)
    return 1 - tanhx ** 2

  def dsigmoid(sigmoidx=None, x=None):
    if sigmoidx is None:
      sigmoidx = sigmoid(x)
    return (1 - sigmoidx) * sigmoidx

  (x, prev_h, prev_c, Wx, Wh, a, sig_ainput, sig_aforget, sig_aoutput, tanh_block_input, tanh_next_c) = cache

  # dnext_h, dnext_c
  dtanh_next_c = dnext_h * sig_aoutput
  dsig_aoutput = dnext_h * tanh_next_c
  dnext_c += dtanh(tanh_next_c) * dtanh_next_c  # dnext_c 共两个来源, backward 参数里有一个, 且 dnext_h 反向推回去也会影响 dnext_c
  dsig_ainput = tanh_block_input * dnext_c
  dtanh_block_input = sig_ainput * dnext_c
  dsig_aforget = prev_c * dnext_c
  dprev_c = sig_aforget * dnext_c

  dainput = dsigmoid(sig_ainput) * dsig_ainput
  daforget = dsigmoid(sig_aforget) * dsig_aforget
  daoutput = dsigmoid(sig_aoutput) * dsig_aoutput
  dblock_input = dtanh(tanh_block_input) * dtanh_block_input

  da = np.hstack((dainput, daforget, daoutput, dblock_input))
  db = np.sum(da, axis=0)
  dWx = x.T.dot(da)
  dx = da.dot(Wx.T)
  dWh = prev_h.T.dot(da)
  dprev_h = da.dot(Wh.T)
  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################

  return dx, dprev_h, dprev_c, dWx, dWh, db













def lstm_forward(x, h0, Wx, Wh, b):
  """
  Forward pass for an LSTM over an entire sequence of data. We assume an input
  sequence composed of T vectors, each of dimension D. The LSTM uses a hidden
  size of H, and we work over a minibatch containing N sequences. After running
  the LSTM forward, we return the hidden states for all timesteps.

  Note that the initial cell state is passed as input, but the initial cell
  state is set to zero. Also note that the cell state is not returned; it is
  an internal variable to the LSTM and is not accessed from outside.

  Inputs:
  - x: Input data of shape (N, T, D)
  - h0: Initial hidden state of shape (N, H)
  - Wx: Weights for input-to-hidden connections, of shape (D, 4H)
  - Wh: Weights for hidden-to-hidden connections, of shape (H, 4H)
  - b: Biases of shape (4H,)

  Returns a tuple of:
  - h: Hidden states for all timesteps of all sequences, of shape (N, T, H)
  - cache: Values needed for the backward pass.
  """
  h, cache = None, None
  #############################################################################
  # TODO: Implement the forward pass for an LSTM over an entire timeseries.   #
  # You should use the lstm_step_forward function that you just defined.      #
  #############################################################################
  N, T, D = x.shape
  N, H = h0.shape
  h = np.zeros((N, T, H))
  next_h = h0
  next_c = np.zeros((N, H))
  cache_dict = dict()
  for t in range(T):
    next_h, next_c, cache_step = lstm_step_forward(x[:, t, :], next_h, next_c, Wx, Wh, b)
    h[:, t, :] = next_h
    cache_dict[t] = cache_step
  cache = x, h0, Wx, Wh, b, cache_dict
  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################

  return h, cache


def lstm_backward(dh, cache):
  """
  Backward pass for an LSTM over an entire sequence of data.]

  Inputs:
  - dh: Upstream gradients of hidden states, of shape (N, T, H)
  - cache: Values from the forward pass

  Returns a tuple of:
  - dx: Gradient of input data of shape (N, T, D)
  - dh0: Gradient of initial hidden state of shape (N, H)
  - dWx: Gradient of input-to-hidden weight matrix of shape (D, 4H)
  - dWh: Gradient of hidden-to-hidden weight matrix of shape (H, 4H)
  - db: Gradient of biases, of shape (4H,)
  """
  dx, dh0, dWx, dWh, db = None, None, None, None, None
  #############################################################################
  # TODO: Implement the backward pass for an LSTM over an entire timeseries.  #
  # You should use the lstm_step_backward function that you just defined.     #
  #############################################################################
  N, T, H = dh.shape
  x, h0, Wx, Wh, b, cache_dict = cache
  N, T, D = x.shape

  dx = np.zeros_like(x)
  dWx = np.zeros_like(Wx)
  dWh = np.zeros_like(Wh)
  db = np.zeros_like(b)
  dprev_h = np.zeros_like(h0)

  dprev_c = np.zeros((N, H))
  for t in reversed(range(T)):
    dxstp, dprev_h, dprev_c, dWxstp, dWhstp, dbstp = \
        lstm_step_backward(dh[:, t, :] + dprev_h, dprev_c, cache_dict[t])
    dx[:, t, :] = dxstp
    dWx += dWxstp
    dWh += dWhstp
    db += dbstp

  dh0 = dprev_h

  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################

  return dx, dh0, dWx, dWh, db






























def temporal_affine_forward(x, w, b):
  """
  Forward pass for a temporal affine layer. The input is a set of D-dimensional
  vectors arranged into a minibatch of N timeseries, each of length T. We use
  an affine function to transform each of those vectors into a new vector of
  dimension M.

  Inputs:
  - x: Input data of shape (N, T, D)
  - w: Weights of shape (D, M)
  - b: Biases of shape (M,)

  Returns a tuple of:
  - out: Output data of shape (N, T, M)
  - cache: Values needed for the backward pass
  """
  N, T, D = x.shape
  M = b.shape[0]
  out = x.reshape(N * T, D).dot(w).reshape(N, T, M) + b
  cache = x, w, b, out
  return out, cache


def temporal_affine_backward(dout, cache):
  """
  Backward pass for temporal affine layer.

  Input:
  - dout: Upstream gradients of shape (N, T, M)
  - cache: Values from forward pass

  Returns a tuple of:
  - dx: Gradient of input, of shape (N, T, D)
  - dw: Gradient of weights, of shape (D, M)
  - db: Gradient of biases, of shape (M,)
  """
  x, w, b, out = cache
  N, T, D = x.shape
  M = b.shape[0]

  dx = dout.reshape(N * T, M).dot(w.T).reshape(N, T, D)
  dw = dout.reshape(N * T, M).T.dot(x.reshape(N * T, D)).T
  db = dout.sum(axis=(0, 1))

  return dx, dw, db


def temporal_softmax_loss(x, y, mask, verbose=False):
  """
  A temporal version of softmax loss for use in RNNs. We assume that we are
  making predictions over a vocabulary of size V for each timestep of a
  timeseries of length T, over a minibatch of size N. The input x gives scores
  for all vocabulary elements at all timesteps, and y gives the indices of the
  ground-truth element at each timestep. We use a cross-entropy loss at each
  timestep, summing the loss over all timesteps and averaging across the
  minibatch.

  As an additional complication, we may want to ignore the model output at some
  timesteps, since sequences of different length may have been combined into a
  minibatch and padded with NULL tokens. The optional mask argument tells us
  which elements should contribute to the loss.

  Inputs:
  - x: Input scores, of shape (N, T, V)
  - y: Ground-truth indices, of shape (N, T) where each element is in the range
       0 <= y[i, t] < V
  - mask: Boolean array of shape (N, T) where mask[i, t] tells whether or not
    the scores at x[i, t] should contribute to the loss.

  Returns a tuple of:
  - loss: Scalar giving loss
  - dx: Gradient of loss with respect to scores x.
  """

  N, T, V = x.shape

  x_flat = x.reshape(N * T, V)
  y_flat = y.reshape(N * T)
  mask_flat = mask.reshape(N * T)

  probs = np.exp(x_flat - np.max(x_flat, axis=1, keepdims=True))
  probs /= np.sum(probs, axis=1, keepdims=True)
  loss = -np.sum(mask_flat * np.log(probs[np.arange(N * T), y_flat])) / N
  dx_flat = probs.copy()
  dx_flat[np.arange(N * T), y_flat] -= 1
  dx_flat /= N
  dx_flat *= mask_flat[:, None]

  if verbose:
    print ('dx_flat: ', dx_flat.shape)

  dx = dx_flat.reshape(N, T, V)

  return loss, dx

