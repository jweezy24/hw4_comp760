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
        # self.w_1 = np.random.normal(scale=0.5, size=(28*28,300))
        self.w_1 = torch.randn((28*28,300)).numpy()*np.sqrt(2/(300))
        # self.w_2 = np.random.normal(scale=0.5, size=(300,200))
        self.w_2 = torch.randn((300,200)).numpy()*np.sqrt(2/ ((200)))
        # self.w_3 = np.random.normal(scale=0.5, size=(200,10))
        self.w_3 = torch.randn((200,10)).numpy()*np.sqrt(2/ ((10) ))

        self.loss_func = nn.CrossEntropyLoss()
        # self.loss_func = cross_entropy_loss


    def feed_forward(self,X):
        inputs = []
        results = []

        inputs.append(X.T)

        o1 = sigmoid(X.T@self.w_1)

        results.append(o1)

        inputs.append(o1)

        o2 = sigmoid(o1@self.w_2)

        results.append(o2)

        inputs.append(o2)
        
        outs = softmax(o2@self.w_3)

        results.append(outs)

        
        return inputs,results
        

    def predict(self,X):




        o1 = sigmoid(X.T@self.w_1)


        o2 = sigmoid(o1@self.w_2)



        outs = softmax(o2@self.w_3)


        preds = torch.argmax(torch.from_numpy(outs), dim=1)
        
        preds = preds.to(torch.float64)

        return preds.numpy()


    def back_pass(self,inputs,results,y,lr,bs):

        y = y.astype(float)
        # print(results[-1])
        err = self.loss_func(torch.from_numpy(results[-1]),torch.from_numpy(y))
        # print(err)
        final = softmax_d(results[-2],y)
        # print(err.shape,final.shape)
        # print((results[-1].T-y))
        out_delta = final
        # print(out_delta)
        # exit()

        # print(final.shape,self.w_3.T.shape)
        l2_bp = sigmoid_p(results[-2])
        l2_err = final@self.w_3.T
        l2_delta = l2_err * l2_bp

        # print(self.w_2.shape,l2_delta.shape)
        l1_bp = sigmoid_p(results[-3])
        l1_err = l2_delta@self.w_2.T

        # print(l1_err.shape , l1_bp.T.shape)
    
        l1_delta = l1_err * l1_bp
        

        # print(inputs[2].T.shape,out_delta.shape)
        self.w_3 -= lr* ((inputs[2].T@out_delta)/bs)
        self.w_2 -= lr*((inputs[1].T@l2_delta)/bs)
        self.w_1 -= lr*((inputs[0].T@l1_delta)/bs)

        return err

        
    def train(self,X,y,lr=0.01,batch=128):
        epoch = 0
        loss = 1
        # y_tensor = torch.from_numpy(y).to(torch.float64)
        while epoch < 10000 or loss < 0.01:
            b = 0
            iters = 0
            running_loss = 0.0
            while (b+1)*batch < X.shape[1]:
                ps = np.random.choice(X.shape[1], batch, replace=False)
                X_batch = X[:,ps]
                y_batch = y[ps]

                # ins,outs = self.feed_forward(X_batch)
                # old_loss = self.back_pass(ins,outs,y_batch,lr,batch)
                # b+=1

                forwards = self.feed_forward_attempt2(X_batch)
                self.back_pass2(forwards,y_batch,lr)
            
                running_loss += 0#old_loss
                
            iters+=b
            preds = self.predict(X)
            print(np.argmax(y,axis=1),preds)
            misses = np.count_nonzero(preds-np.argmax(y,axis=1))
            print(y.shape)
            loss = np.count_nonzero(preds-np.argmax(y,axis=1))/y.shape[0]
            err =running_loss/b
            print(f" Loss:{err} Epoch: {epoch} Iterations: {iters} MSE: {loss} Misses:{misses}")
            epoch += 1


if __name__ == "__main__":
    from sklearn import preprocessing
    ohe = preprocessing.OneHotEncoder()

    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
        
    

    # X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=42)
    X_train = train_images.reshape(-1,train_images.shape[0])
    X_test = test_images.reshape(-1,test_images.shape[0])
    
    # X_train = X_train.astype(float)
    X_train = X_train/255
    y_train = train_labels.reshape(-1, 1)
    ohe.fit(y_train)
    transformed_train = ohe.transform(y_train).toarray()
    

    dnn = DNN()
    dnn.train(X_train,transformed_train)

