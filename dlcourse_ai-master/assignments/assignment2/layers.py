import numpy as np
from linear_classifer import softmax, cross_entropy_loss


def l2_regularization(W, reg_strength):
    """
    Computes L2 regularization loss on weights and its gradient

    Arguments:
      W, np array - weights
      reg_strength - float value

    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
    """
    # TODO: Copy from the previous assignment
    #raise Exception("Not implemented!")
    
    loss = (W*W).sum() * reg_strength
    grad = 2 * W * reg_strength
    
    return loss, grad


def softmax_with_cross_entropy(predictions, target_index):
    """
    Computes softmax and cross-entropy loss for model predictions,
    including the gradient

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss, single value - cross-entropy loss
      dprediction, np array same shape as predictions - gradient of predictions by loss value
    """
    # TODO: Copy from the previous assignment
    #raise Exception("Not implemented!")
    
    predictions = predictions.copy()

    probs = softmax(predictions)

    loss = cross_entropy_loss(probs, target_index).mean()

    mask = np.zeros_like(predictions)
    mask[np.arange(len(mask)), target_index] = 1
    # mask[target_index] = 1

    dprediction = - (mask - softmax(predictions)) / mask.shape[0]


    return loss, dprediction


class Param:
    """
    Trainable parameter of the model
    Captures both parameter value and the gradient
    """

    def __init__(self, value):
        self.value = value
        self.grad = np.zeros_like(value)


class ReLULayer:
    def __init__(self):
        self.X = None

    def forward(self, X):
        # TODO: Implement forward pass
        # Hint: you'll need to save some information about X
        # to use it later in the backward pass
        result = np.maximum(0, X)
        self.X = X
        
        return result

    def backward(self, d_out):
        """
        Backward pass

        Arguments:
        d_out, np array (batch_size, num_features) - gradient
           of loss function with respect to output

        Returns:
        d_result: np array (batch_size, num_features) - gradient
          with respect to input
        """
        # TODO: Implement backward pass
        # Your final implementation shouldn't have any loops
        d_result = (self.X>0)*d_out
        
        return d_result

    def params(self):
        # ReLU Doesn't have any parameters
        return {}


class FullyConnectedLayer:
    def __init__(self, n_input, n_output):
        self.W = Param(0.001 * np.random.randn(n_input, n_output))
        self.B = Param(0.001 * np.random.randn(1, n_output))
        self.X = None

    def forward(self, X):
        # TODO: Implement forward pass
        # Your final implementation shouldn't have any loops
#         raise Exception("Not implemented!")
        W = self.W.value
        B = self.B.value
        self.X = Param(X)
        out = np.dot(X,W)+B
        
        return out
           

    def backward(self, d_out):
        """
        Backward pass
        Computes gradient with respect to input and
        accumulates gradients within self.W and self.B

        Arguments:
        d_out, np array (batch_size, n_output) - gradient
           of loss function with respect to output

        Returns:
        d_result: np array (batch_size, n_input) - gradient
          with respect to input
        """
        # TODO: Implement backward pass
        # Compute both gradient with respect to input
        # and gradients with respect to W and B
        # Add gradients of W and B to their `grad` attribute

        # It should be pretty similar to linear classifier from
        # the previous assignment
        # raise Exception("Not implemented!")
        W = self.W.value
        X = self.X.value
        
        dW = np.dot(X.T, d_out)
        dB = np.dot(np.ones((X.shape[0],1)).T, d_out)
        dX = np.dot(d_out, W.T)
        
        self.W.grad += dW
        self.B.grad += dB
        

        return dX

    def params(self):
        return {'W': self.W, 'B': self.B}
