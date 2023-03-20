import numpy as np

#Tanh translation
def tanh(x,w):
    return np.tanh(w.T@x)#(np.exp(w.T@x) - np.exp(-w.T@x))/(np.exp(w.T@x)+np.exp(-w.T@x))

#tanh derivative
def tanh_p(x,w):
    res = tanh(x, w)
    # print(res.shape)
    return 1- np.square(res)