import numpy as np
import torch.nn as nn
import tensorflow as tf
import torch
from relu import *
from sigmoid import *
from tanh import *
from softmax import *
from loss_funcs import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class DNN:
    def __init__(self):
        self.w_1 = np.random.normal(scale=0.5, size=(28*28,300))
        self.w_2 = np.random.normal(scale=0.5, size=(300,200))
        self.w_3 = np.random.normal(scale=0.5, size=(200,10))

        # self.loss_func2 = nn.CrossEntropyLoss()
        self.loss_func = cross_entropy_loss


    def feed_forward(self,X):
        inputs = []
        results = []

        inputs.append(X.T)

        o1 = sigmoid(self.w_1.T@X.T)

        results.append(o1)

        inputs.append(o1)

        o2 = sigmoid(self.w_2.T@o1)

        results.append(o2)

        inputs.append(o2)

        outs = softmax(self.w_3.T@o2)

        results.append(outs)

        inputs.append(outs)
        
        return inputs,results
    
    def predict(self,X):
        inputs = []
        results = []
        X = X.T
        inputs.append(X.T)

        o1 = sigmoid(self.w_1.T@X.T)

        results.append(o1)

        inputs.append(o1)

        o2 = sigmoid(self.w_2.T@o1)

        results.append(o2)

        inputs.append(o2)

        outs = softmax(self.w_3.T@o2)

        results.append(outs)

        inputs.append(outs)
        
        preds = torch.argmax(torch.from_numpy(results[-1]), dim=0)
        
        preds = preds.to(torch.float64)

        return preds


    def back_pass(self,inputs,results,y,lr,bs):
        preds = self.predict(inputs[0]).to(torch.float64).cuda()
        y_tensor = torch.from_numpy(y).to(torch.float64).cuda()
        
        # print(preds,y_tensor)
        
        err =self.loss_func(y_tensor.cpu().numpy(),preds.cpu().numpy())/(bs*10)
        # print(err)
        out_bp = softmax_d(results[-1],y)
        out_delta = err*out_bp

        l2_bp = sigmoid_p(results[-2])
        l2_err = self.w_3@out_delta
        l2_delta = l2_err * l2_bp

        l1_bp = sigmoid_p(results[-3])
        l1_err = self.w_2@l2_delta
        l1_delta = l1_err * l1_bp
        
        # print(y.size)
        self.w_3 -= lr* ((inputs[2]@out_delta.T))
        self.w_2 -= lr*((inputs[1]@l2_delta.T))
        self.w_1 -= lr*((inputs[0]@l1_delta.T))

        
    def train(self,X,y,lr=0.001,batch=64):
        epoch = 0
        loss = 1
        y_tensor = torch.from_numpy(y).to(torch.float64)
        while epoch < 10000 or loss < 0.01:
            b = 0
            iters = 0
            running_loss = 0.0
            while (b+1)*batch < X.shape[0]:
                ps = np.random.choice(X.shape[0], batch, replace=False)
                X_batch = X[ps,:]
                y_batch = y[ps]

                ins,outs = self.feed_forward(X_batch)
                self.back_pass(ins,outs,y_batch,lr,batch)
                b+=1
                
                ins = self.predict(X_batch.T).to(torch.float64).numpy()
                y_batch = torch.from_numpy(y_batch).to(torch.float64).numpy()
                running_loss += self.loss_func(ins,y_batch)/(batch*10)
                
            iters+=b
            # preds = self.predict(X.T)
            # loss = np.count_nonzero(preds-y)/y.size
            err =running_loss/b
            print(f"Loss: {err} Iterations: {iters}")
            epoch += 1
if __name__ == "__main__":


    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
        
    

    # X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=42)
    X_train = train_images.reshape(train_images.shape[0], -1)
    X_test = test_images.reshape(test_images.shape[0], -1)
    

    dnn = DNN()
    dnn.train(X_train,train_labels)

