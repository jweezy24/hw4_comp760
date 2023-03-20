import numpy as np

#Simple implementation of sigmoid
def sigmoid(x):
    x = np.clip( x, -600, 600 )
    return (1/(1+ np.exp(-1*(x))))

#Derivative of sigmoid
def sigmoid_p(x):
    res = sigmoid(x)

    # for i in range(res.shape[1]):
    #     res[:,i] = res[:,i]*(1-res[:,i])
    res = sigmoid(x)*(1-sigmoid(x))
    return res

