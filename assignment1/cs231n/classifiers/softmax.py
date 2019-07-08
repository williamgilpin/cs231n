import numpy as np
from random import shuffle
from past.builtins import xrange

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
  

  for yi, Xi in zip(y, X):
    
    # find unnormed probs and use trick from 
    # lecture for numeric stability
    fi = Xi.dot(W)
    fi -= np.max(fi) 

    sum_val = 0.0
    for j, fij in enumerate(fi):
        sum_val += np.exp(fij)
        if yi==j:
            dW[:,j] += (np.exp(fij)/np.sum(np.exp(fi))-1)*Xi
        else:
            dW[:,j] += (np.exp(fij)/np.sum(np.exp(fi)))*Xi
        
    Li = -fi[yi] + np.log(sum_val)
    loss += Li
    
  loss /= len(X)
  loss += reg*np.sum(W*W)
    
  dW /= len(X)
  dW += 2*reg*W
    
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
  
  f = np.dot(X,W)
  f -= np.max(f,axis=1)[:,np.newaxis]
  sum_val = np.sum(np.exp(f), axis=1)[:,np.newaxis]
  yij= np.exp(f)/sum_val

  loss = np.sum(-np.log(yij[range(y.shape[0]), y]))

  grad_array = np.exp(f)/sum_val
  grad_array[range(y.shape[0]),y] -= 1 
  dW = np.dot(grad_array.T, X).T

  loss /= len(X)
  loss += reg*np.sum(W*W)
    
  dW /= len(X)
  dW += 2*reg*W 
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

