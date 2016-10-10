import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

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
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in range(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]

    margin_positive_count = 0  # count 所有 margin>0 的次数

    for j in range(num_classes):
      if j == y[i]:
        continue  # j==y[i]时 在微分的两个part里都不会对dW有贡献
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        margin_positive_count += 1 # 没满足边界值的分类数量 对损失函数产生贡献
        loss += margin
        dW[:, j] += X[i]

    dW[:, y[i]] -= X[i] * margin_positive_count

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  loss += 0.5 * reg * np.sum(W * W)  # Add regularization to the loss.

  dW /= num_train
  dW += reg * W        # 需要加上正则化损失的求导, 但还不太明白

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################
  return loss, dW




def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  # num_classes = W.shape[1]
  num_train = X.shape[0]
  total_scores = X.dot(W)

  correct_class_scores = total_scores[range(num_train), y].reshape(-1, 1)
  margins = total_scores - correct_class_scores + 1
  margins[range(num_train), y] = 0  # 正确的分类之前也算成有1的损失, 需要 = 0
  margins = margins.clip(min=0)
  loss = margins.sum() / num_train

  loss += 0.5 * reg * np.sum(W * W)  # Add regularization to the loss.

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################

  # 首先假设所有的分类全都是不正确的, 算出一个 dW
  # 以 j == yi 做一个mask, 按照 mask 过滤出来, 再修改 j == yi 的对应梯度

  # print('after loss\n', margins)
  margins[margins > 0] = 1  # 处理为仅含 0 1 值, 即 都按照 j != yi 来算
  # print('after 0 1\n', margins)

  margins[range(num_train), y] = -margins.sum(axis=1)  # 修正 j == yi 的情况
  # print('after sum neg yi\n', margins)

  dW = X.T.dot(margins)
  dW /= num_train
  dW += reg * W
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
