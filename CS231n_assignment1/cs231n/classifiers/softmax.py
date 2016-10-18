import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]

  for i in range(num_train):
    scores = X[i].dot(W)
    # correct_class_score = scores[y[i]]
    yi = y[i]
    scores -= np.max(scores)  # 最大分数归为0
    probs = np.exp(scores) / np.sum(np.exp(scores)) # 归一化概率
    loss += -np.log(probs[yi])

    # 需要 Li(w) = -log(e^(wyi*x) / ∑(e^(wj*x))) 对 w 求导
    # 其中 wyi 表示正确分类的得分
    #      wj 表示所有分类的得分, 只用于∑ (含正确的那个分类)
    #      k 表示单指j其中的某一个, 可以是正确分类或错误分类

    # 设 f(k) = wk*x
    # 设 P(k) = e^(wk*x) / ∑(e^(wj*x)   即'归一化概率',
    #                                   可事先减去最大的分值, 使最大数归0方便计算

    # 需要求 dLi/dw
    # 分解为 dLi/dw = dLi/df(k) * df(k)/dw
    # part1 : 求 dLi/df
    # L(f) = -log(e^f / ∑j(e^f))

    # 当 k 为正确分类时:
    # L(fk) = -log(e^fk / ∑(e^fj))
    #       = -fk + log(∑(e^fj))   ---- log(a/b) = loga - logb
    # dL/dfk = -1 + (1/(∑(e^fj))) * (∑(e^fj))'   ---- log(x)' = 1/x
    #        = -1 + (1/(∑(e^fj))) * (e^fk)'      ---- 因 ∑(e^fj) 中只有 fk 那一项有用
    #                                            ---- 其余是常量项
    #        = -1 + (1/(∑(e^fj))) * (e^fk) * (fk)' ---- e^x' = e^x
    #        = -1 + P(k)

    # 当 k 为错误分类时:
    # L(fk) = -log(e^fyi / ∑(e^fj))
    #       = -fyi + log(∑(e^fj))   ---- log(a/b) = loga - logb
    # dL/dfk = 0 + (1/(∑(e^fj))) * (∑(e^fj))'
    # 同上 ....
    # dL/dfk = P(k)

    # part2 : 求 df(k)/dw -> 比较简单 这个就是 x

    # 最终:
    # dLi/dw = dLi/df(k) * df(k)/dw
    # = (-1 + P(k)) * x  (当 k 为正确分类 k = yi)
    #   (P(k)) * x       (当 k 为错误分类 k ≠ yi)

    for j in range(num_classes):
      if j == yi:
        dW[:, j] += (-1 + probs[j]) * X[i]
      else:
        dW[:, j] += (probs[j]) * X[i]

  loss /= num_train
  loss += 0.5 * reg * np.sum(W * W)  # Add regularization to the loss.
  dW /= num_train
  dW += reg * W

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################
  return loss, dW






def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.
  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0]
  Scores = X.dot(W)
  Scores -= np.max(Scores, axis=1).reshape(num_train, 1)

  Sum_e_scores = np.sum(np.exp(Scores), axis=1)
  Probs = np.exp(Scores) / Sum_e_scores.reshape(num_train, 1)
  # Mask = np.zeros_like(Probs)
  # Mask[range(num_train), y] = 1

  loss = np.sum(-np.log(Probs[range(num_train), y]))
  Probs[range(num_train), y] -= 1
  dW = X.T.dot(Probs)

  loss /= num_train
  loss += 0.5 * reg * np.sum(W * W)
  dW /= num_train
  dW += reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

