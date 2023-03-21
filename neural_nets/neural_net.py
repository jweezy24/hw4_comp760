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

        inputs.append(X)

        o1 = sigmoid(self.w_1.T@X)

        results.append(o1)

        inputs.append(o1)

        o2 = sigmoid(self.w_2.T@o1)

        results.append(o2)

        inputs.append(o2)

        outs = softmax(self.w_3.T@o2)

        results.append(outs)

        
        return inputs,results
    
    def predict(self,X):
        inputs = []
        results = []
        # X = X.T
        inputs.append(X)

        o1 = sigmoid(self.w_1.T@X)

        results.append(o1)

        inputs.append(o1)

        o2 = sigmoid(self.w_2.T@o1)

        results.append(o2)

        inputs.append(o2)

        outs = softmax(self.w_3.T@o2)

        results.append(outs)

        inputs.append(outs)
        
        # print(results[-1])
        preds = torch.argmax(torch.from_numpy(results[-1]), dim=0)
        
        preds = preds.to(torch.float64)

        return preds


    def back_pass(self,inputs,results,y,lr,bs):
        # preds = self.predict(inputs[0]).to(torch.float64).cuda()
        # y_tensor = torch.from_numpy(y).to(torch.float64).cuda()
        
        y = y.astype(float)
        
        err =(results[-1].T-y)
        # print(err)
        final = softmax_derivative(results[-1])
        # print(err.shape,final.shape)
        # print((results[-1].T-y))
        out_delta = err*final.T
        # print(out_delta)
        # exit()

        # print(self.w_3.shape,final.shape)
        l2_bp = sigmoid_p(results[-2])
        l2_err = self.w_3@final
        l2_delta = l2_err * l2_bp

        # print(self.w_2.shape,l2_delta.shape)
        l1_bp = sigmoid_p(results[-3])
        l1_err = self.w_2@l2_delta

        # print(l1_err.shape , l1_bp.T.shape)
    
        l1_delta = l1_err * l1_bp
        
        # print(inputs[2].shape,out_delta.T.shape)
        self.w_3 -= lr* ((inputs[2]@out_delta)/bs)
        self.w_2 -= lr*((inputs[1]@l2_delta.T)/bs)
        self.w_1 -= lr*((inputs[0]@l1_delta.T)/bs)
        return err

        
    def train(self,X,y,lr=0.01,batch=32):
        epoch = 0
        loss = 1
        y_tensor = torch.from_numpy(y).to(torch.float64)
        while epoch < 10000 or loss < 0.01:
            b = 0
            iters = 0
            running_loss = 0.0
            while (b+1)*batch < X.shape[1]:
                ps = np.random.choice(X.shape[1], batch, replace=False)
                X_batch = X[:,ps]
                y_batch = y[ps]

                ins,outs = self.feed_forward(X_batch)
                old_loss = self.back_pass(ins,outs,y_batch,lr,batch)
                b+=1
                
            
                running_loss += old_loss
                
            iters+=b
            preds = self.predict(X)
            print(np.argmax(y,axis=1),preds)
            misses = np.count_nonzero(preds-np.argmax(y,axis=1))
            print(y.shape)
            loss = np.count_nonzero(preds-np.argmax(y,axis=1))/y.shape[0]
            err =running_loss/b
            print(f" Epoch: {epoch} Iterations: {iters} MSE: {loss} Misses:{misses}")
            epoch += 1
if __name__ == "__main__":
    from sklearn import preprocessing
    ohe = preprocessing.OneHotEncoder()

    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
        
    

    # X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=42)
    X_train = train_images.reshape(-1,train_images.shape[0])
    X_test = test_images.reshape(-1,test_images.shape[0])
    
    X_train = X_train/255
    y_train = train_labels.reshape(-1, 1)
    ohe.fit(y_train)
    transformed_train = ohe.transform(y_train).toarray()
    

    dnn = DNN()
    dnn.train(X_train,transformed_train)

