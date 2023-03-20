import numpy as np

def softmax(x):
    x = np.clip( x, -600, 600 )
    exp = np.exp(x)
    ret = exp / np.sum(exp, axis=0, keepdims=True)
    return ret

def softmax_torch(x):
    import torch
    scores = torch.from_numpy(x)
    probs = torch.nn.functional.softmax(scores, dim=0)
    return probs.numpy()

def softmax_d(x,y):
    m = y.shape[0]
  
    grad = softmax(x).T

    grad[range(m),y] -=1

    grad = grad/m
    return grad.T