import numpy as np

def softmax(x):
    x = np.clip( x, -600, 600 )
    exp = np.exp(x)
    ret = exp / np.sum(exp, axis=1, keepdims=True)
    return ret
    
def softmax2(z):
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z,axis=0)

def softmax_torch(x):
    import torch
    scores = torch.from_numpy(x)
    probs = torch.nn.functional.softmax(scores, dim=0)
    return probs.numpy()

def softmax_d2(x):
        # Numerically stable with large exponentials
        x = np.clip( x, -600, 600 )
        exps = np.exp(x - x.max())
        return 1 - exps / np.sum(exps, axis=0)

def softmax_derivative(z):
    s = z  # apply the softmax function to z
    d = np.zeros_like(s)  # initialize the Jacobian matrix with zeros
    for i in range(s.shape[1]):  # for each data point
        s_i = s[:, i].reshape(-1, 1)  # reshape to a column vector
        d_i = np.diagflat(s_i) - np.dot(s_i, s_i.T)  # compute the Jacobian for the i-th point
        d[:, i] = d_i @ z[:, i]  # multiply by the input features and store in the output matrix
    return d

def softmax_derivative2(z):
    """
    Compute the derivative of the softmax function for each row of a matrix z.

    Arguments:
    z -- A numpy array of shape (m, n)

    Returns:
    A numpy array of shape (m, n) containing the derivative of the softmax function for each input point's output vector
    """
    # Compute the softmax values for each row of the matrix
    softmax_z = z

    # Compute the derivative of the softmax function
    softmax_derivative_z = softmax_z * (1 - softmax_z)

    return softmax_derivative_z

def softmax_derivative3(x):
    s = x
    d = np.zeros((s.shape[0], s.shape[0], s.shape[1]))
    for i in range(s.shape[1]):
        d[:, :, i] = np.diag(s[:, i]) - (s[:, i]@s[:, i].T)
    return d

def softmax_d(x,y):
    m = y.shape[0]
    # print(m)
    # print(x)
    grad = x.T

    # print(grad)

    grad[range(m),np.argmax(y)] -=1
    grad = 1-grad
    grad = grad
    
    return grad.T