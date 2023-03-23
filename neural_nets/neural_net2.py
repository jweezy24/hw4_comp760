import numpy as np
import torch.nn as nn
import tensorflow as tf
import torch
import matplotlib.pyplot as plt

def sigmoid(z):
    s = np.zeros_like(z)
    idx = z > 0
    s[idx] = 1 / (1 + np.exp(-z[idx]))
    s[~idx] = np.exp(z[~idx]) / (1 + np.exp(z[~idx]))
    return s

def softmax(z):
    max_z = np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z - max_z)
    sum_exp_z = np.sum(exp_z, axis=1, keepdims=True)
    return exp_z / sum_exp_z

def forward(X, W1, W2, W3):
    Z1 = np.dot(W1, X.T)
    H1 = sigmoid(Z1)
    Z2 = np.dot(W2, H1)
    H2 = sigmoid(Z2)
    Z3 = np.dot(W3, H2)
    Y_hat = softmax(Z3)
    return Y_hat.T,(Z1,Z2,Z3),(X,H1,H2)

def loss_func(X, Y, W1, W2, W3):
    Y_hat,_,_ = forward(X, W1, W2, W3)
    m = Y.shape[0]
    log_probs = -np.log(Y_hat[range(m), np.argmax(Y, axis=1)])
    loss = np.sum(log_probs) / m
    return loss

def backward(X, Y, W1, W2, W3):
    m = X.shape[0]
    Y_hat,Zs,Hs = forward(X, W1, W2, W3)
    Z1,Z2,Z3 = Zs
    _,H1,H2 = Hs 

    Delta3 = Y_hat - Y
    Delta2 = np.dot(W3.T, Delta3.T) * sigmoid(Z2) * (1 - sigmoid(Z2))
    Delta1 = np.dot(W2.T, Delta2) * sigmoid(Z1) * (1 - sigmoid(Z1))
    Grad_W3 = np.dot(Delta3.T, H2.T)/m
    Grad_W2 = np.dot(Delta2, H1.T)/m
    Grad_W1 = np.dot(Delta1, X)/m
    # print(Grad_W1,Grad_W2,Grad_W3)
    # print(Delta1,X)
    return Grad_W1, Grad_W2, Grad_W3

def train(X,y,x_test,y_test,lr=0.0002,batch=128,epochs=150):
    w_1 = np.random.randn(300, 784) * np.sqrt(1/784)
    w_2 = np.random.randn(200, 300) * np.sqrt(1/300)
    w_3 = np.random.randn(10, 200) * np.sqrt(1/200)

    #Init weights to zeros
    w_1 = np.zeros((300, 784)) 
    w_2 = np.zeros((200, 300)) 
    w_3 = np.zeros((10, 200)) 
    
    epoch = 0
    loss = 1
    losses = []
    losses_test = []
    test_errors = []
    iters = 0
    # y_tensor = torch.from_numpy(y).to(torch.float64)
    while epoch < epochs:
        b = 0
        running_loss = 0.0
        while (b+1)*batch < X.shape[0]:
            ps = np.random.choice(X.shape[0], batch, replace=False)
            X_batch = X[ps,:]
            y_batch = y[ps]

            gw1,gw2,gw3 = backward(X_batch,y_batch,w_1,w_2,w_3)
            
            w_1 += -lr*gw1
            w_2 += -lr*gw2
            w_3 += -lr*gw3
            
            running_loss += loss_func(X_batch,y_batch,w_1,w_2,w_3)
            b+=1
            
        iters+=b
        preds = np.argmax(forward(X,w_1,w_2,w_3)[0],axis=1)
        preds_test = np.argmax(forward(x_test,w_1,w_2,w_3)[0],axis=1)
        test_loss = loss_func(x_test,y_test,w_1,w_2,w_3)
        losses_test.append(test_loss)
        print(np.argmax(y,axis=1),preds)
        misses = np.count_nonzero(preds-np.argmax(y,axis=1))
        misses_test = np.count_nonzero(preds_test-np.argmax(y_test,axis=1))
        test_errors.append(misses_test)
        loss = np.count_nonzero(preds-np.argmax(y,axis=1))/y.shape[0]
        err =running_loss/b
        losses.append(err)
        print(f" Loss:{err} Epoch: {epoch} Iterations: {iters} MSE: {loss} Misses:{misses}")
        epoch += 1

    return w_1,w_2,w_3,losses,losses_test,test_errors

if __name__ == "__main__":
    from sklearn import preprocessing
    ohe = preprocessing.OneHotEncoder()

    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
        
    epochs = 15

    # X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=42)
    X_train = train_images.reshape(train_images.shape[0],-1)
    X_test = test_images.reshape(test_images.shape[0],-1)
    
    # X_train = X_train.astype(float)
    X_train = X_train/255
    y_train = train_labels.reshape(-1, 1)
    y_test = test_labels.reshape(-1,1)
    ohe.fit(y_train)
    transformed_train = ohe.transform(y_train).toarray()
    ohe.fit(y_test)
    transformed_test = ohe.transform(y_test).toarray()
    
    w1,w2,w3,losses_train,losses_test,misses_test = train(X_train,transformed_train,X_test,transformed_test,epochs=epochs)
    

    plt.title("Learning Curve")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.plot(range(epochs),losses_train,label="Training")
    plt.legend()
    plt.savefig("learning_curve.pdf")

    plt.clf()

    plt.title("Test Errors as a Function of Epochs ")
    plt.xlabel("Epochs")
    plt.ylabel("Errors")
    plt.plot(range(epochs),misses_test)
    plt.savefig("error_rates.pdf")

    