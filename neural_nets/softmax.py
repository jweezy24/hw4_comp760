import numpy as np

def softmax(x):
    # x = np.clip( x, -600, 600 )
    # print(x.shape)
    exp = np.exp(x)
    
    ret = exp / np.sum(exp, axis=1, keepdims=True)
   
    return ret
    
def softmax2(z):
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z,axis=0)

def softmax_torch(x):
    import torch
    scores = torch.from_numpy(x)
    probs = torch.nn.functional.softmax(scores, dim=1)
    return probs.numpy()

def softmax_d(x,y):
    y_i = y.astype(int)
    # print(x.shape)
    grad = softmax(x)
    # print(grad[range(y_i.shape[0]),np.argmax(y_i,axis = 1)],x.shape,y.shape)
    c = 0
    for i in np.argmax(y_i,axis = 1):
        grad[c,i] -= 1
        c+=1
    
    return grad

def softmax_d2(x,y):
    y_i = y.astype(int)
    # print(x.shape)
    grad = x
    # print(grad[range(y_i.shape[0]),np.argmax(y_i,axis = 1)],x.shape,y.shape)
    c = 0
    for i in np.argmax(y_i,axis = 1):
        grad[i,c] -= 1
        c+=1
    
    return grad
