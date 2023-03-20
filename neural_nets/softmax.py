import numpy as np

def softmax(x):
    x = np.clip( x, -600, 600 )
    exp = np.exp(x - np.max(x))
    return exp / np.sum(exp, axis=0, keepdims=True)

def softmax_torch(x):
    import torch
    scores = torch.from_numpy(x)
    probs = torch.nn.functional.softmax(scores, dim=0)
    return probs.numpy()

def softmax_d(x,y):
    m = y.shape[0]
    grad = softmax(x)
    grad[y,range(m)] -=1
    grad = grad/m
    # print(grad.T,grad.T.shape)
    # exit()
    return grad