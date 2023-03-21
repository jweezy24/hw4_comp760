
import numpy as np

def log_loss(y,y_hat):
    y = y.flatten()
    y_hat = y_hat.flatten()
    print(y,y_hat)
    print(y==y_hat)
    # Avoid numerical instability by clipping the predicted probabilities
    y_hat = np.clip(y_hat, 1e-15, 1 - 1e-15)
    
    # Compute cross-entropy loss
    loss = -np.sum(y * np.log(y_hat))
    print(loss, y==y_hat)
    return loss

def softmax2(x):
    exp = np.exp(x)
    return exp / np.sum(exp, axis=0, keepdims=True)

def cross_entropy_loss(y_true, y_pred):
    # print(y_true.shape,y_pred.shape)
    p = np.log(y_pred)
    # print(y_true.shape,p.shape)
    # print(y_true.T * p)
    loss = -np.sum(y_true.T * p,axis=0)
    # print(loss) 
    return loss

def mse_loss(y,y_hat):
    y_hat = y_hat.flatten()
    y = y.flatten()
    return (1/y.size) * ((y-y_hat)**2).sum()