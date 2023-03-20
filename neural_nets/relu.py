import numpy as np

#ReLU implementation
def relu(x,w):

    return np.maximum(0,w.T@x)

#psudo-derivative of ReLU
def relu_p(x,w):
    res = relu(x, w)
    res[res > 0] = 1
    res[res <= 0 ] = 0
    return res