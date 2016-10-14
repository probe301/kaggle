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
    scores -= np.max(scores)  # 最大分数归为0
    p = - np.log(np.exp(scores[y[i]]) / np.sum(np.exp(scores)))
    loss += p

    # wyi and wj (where j!=yi)
    # f(wyi) => log(e^(x*wyi)+b) - x*wyi  ---- where b is sum(wj*x)
    # d/dwyi => x*(e^(x*wyi)) / (e^(x*wyi)+b) - x
    # f(wj) => log(e^(x*wj + x*wyi + x*wk) - x*wyi)    ---- where k is not j, not yi
    # d/dwj => 跟上面差不多, 甚至最后的 - x*wyi 不是自变量了, 还能少一项

    # 可能不需要分别对待 wyi 和 wj, 无所谓正确分类和错误分类, 应该一视同仁计算
    # loss = - log(e^(wyi*x) / (e^(wyi*x) + sum(e^(wj*x))))
    # dloss/dwyi (yi可以表示正确或错误分类, 与SVM不同)
    # = - x*b / (e^(x*wyi) + b)   ---- where b = sum(e^(x*wj)) < j!=yi
    #
    for j in range(num_classes):
      b = np.sum(np.exp(scores[[k for k in range(num_classes) if k != j]]))
      dW[:, j] = - X[i] * b / (np.exp(scores[y[i]]) + b)

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
  pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

