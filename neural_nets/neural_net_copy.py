import numpy as np
import torch.nn as nn
import tensorflow as tf
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from relu import *
from sigmoid import *
from tanh import *
from softmax import *
from loss_funcs import *


class HiddenLayer:
    def __init__(self,alg,nodes,layer_num=0):
        self.alg = alg
        self.nodes = nodes
        self.layer_num = layer_num
        if alg.lower() == "relu":
            self.activation_function = relu
            self.derivative_func = relu_p
        elif alg.lower() == "sigmoid":
            self.activation_function = sigmoid
            self.derivative_func = sigmoid_p
        elif alg.lower() == "tanh":
            self.activation_function = tanh
            self.derivative_func = tanh_p
        elif alg.lower() == "softmax":
            self.activation_function = softmax
            self.derivative_func = softmax_d
        else:
            raise("Activation function not supported.")
    
    def forward_pass(self,x,w):
        res = self.activation_function(x,w)
        return res
    
    def backward_pass(self,x):
        I = np.identity(len(x))
        res = self.derivative_func(x,I)
        return res
    def __str__(self):
        return f"Hidden Layer with {self.nodes} nodes, {self.alg} activation function, and is the {self.layer_num} layer."

class NeuralNet:
    def __init__(self,layers,algs,node_sizes,input_size,loss="log"):
        #set loss function
        if loss == "log":
            self.loss_func = log_loss
        elif loss == "mse":
            self.loss_func = nn.L1Loss()
        elif loss == "ce":
            self.loss_func = nn.CrossEntropyLoss()
        
        #Initialize weights 
        self.ws = []
        #simple layers check
        self.layers = []
        if layers != len(algs)+1:
            raise("Need to assign an activation function per layer")

        self.layers.append(HiddenLayer(algs[0],node_sizes[0],layer_num=0))
        w_new = np.random.normal(scale=0.5, size=(input_size,node_sizes[0])) 
        self.ws.append(w_new)
        #Create layers  
        for i in range(len(algs)-1):
            print(algs[i])
            self.layers.append(HiddenLayer(algs[i+1],node_sizes[i+1],layer_num=i+1))
            w_new = np.random.normal(scale=0.5, size=(node_sizes[i],node_sizes[i+1])) 
            self.ws.append(w_new)
        
        self.layers.append(HiddenLayer(algs[-1],node_sizes[-1],i+1))
        self.ws.append(np.random.normal(scale=0.5, size=(node_sizes[-1],node_sizes[-1])) )

    
    def loss(self,y,y_hat):
        return self.loss_func(y, y_hat)
         

    def back_propagation(self,X,results,inputs,y,lr):
        # print(results[-1].shape,y.shape)
        preds = torch.argmax(torch.from_numpy(results[-1]), dim=1)
        
        preds = preds.to(torch.float64)

        y_tensor = torch.from_numpy(y[0])
        res_tensor = torch.from_numpy(results[-1])

        print(preds,y_tensor)
        out_err = self.loss(y_tensor,preds)
        out_err = out_err.numpy()
        # print(out_err)

        
        out_bp = (self.layers[-1].backward_pass(results[-1])) 
        # print(out_err.shape,out_bp.shape)
        out_delta = out_err * out_bp
        # out_delta = out_delta.reshape((1,-1))
        # print(out_delta.shape)
        deltas = [ out_delta ]
        for i in range(len(self.layers)-2,-1,-1):
            # print(i)
            layer_bp = self.layers[i].backward_pass(results[i])
            cur_ws = self.ws[i+1]

            # print(cur_ws.shape,deltas[-1].shape)
            hlayer_error = cur_ws@deltas[-1]

            # print(hlayer_error.shape,layer_bp.shape)
            new_delta = hlayer_error * layer_bp

            # print(new_delta.shape)
            deltas.append(new_delta)
        
        c=0
        for i in range(len(self.ws)-1,-1,-1):
            cur_ws = self.ws[i]
            # print(cur_ws.shape)
            # print(deltas[c].shape)
            # print(i)
            if i > 0:
                # print(results[i-1].shape,deltas[c].T.shape)
                w_up = (results[i-1] @ deltas[c].T)/y.size
                c+=1
            else:
                # print(X.T.shape,deltas[c].shape)
                w_up = (X.T @ deltas[c].T)/y.size

            
            # print( (cur_ws - lr*w_up) )
            self.ws[i] = cur_ws - lr*w_up
             


    def node_error(self,results,y):
        node_errs = []
        for node in results:
            ave_mse = 0
            losses = node-y
            losses = losses.flatten()
            for l in losses:
                ave_mse+=l**2
            ave_mse = ave_mse/len(losses)
            node_errs.append(ave_mse)
        return node_errs

    def forward_feed(self,X):

        c = 0
        res_results = []
        inputs = []
        for layer in self.layers:
            if c == 0:
                inputs.append(X.T)
                res = layer.forward_pass(X.T, self.ws[c])
                res_results.append(res)
            else:
                inputs.append(res_results[c-1])
                res = layer.forward_pass(res_results[c-1], self.ws[c])
                res_results.append(res)

            c+=1
            
        return inputs,res_results

    def train(self,X,y,batch=10,lr=0.1):
        y = y.reshape((1,-1))
        y = y.astype(float)
        epochs = 0
        loss = 1
        y_tensor = torch.from_numpy(y)
        while loss > 0.001:
            b = 0
            while b*batch < X.shape[0]: 
                X_batch = X[b*batch:(b+1)*batch,:]
                y_batch = y[:,b*batch:(b+1)*batch]
                inputs,results = self.forward_feed(X_batch)
                self.back_propagation(X_batch,results,inputs,y_batch,lr)
                # print(results[-1].shape)
                b+=1
            inputs,results = self.forward_feed(X)
            # print(results[-1])
            # print(y)
            preds = torch.argmax(torch.from_numpy(results[-1]), dim=1)
        
            preds = preds.to(torch.float64)
            # print(self.ws)
            loss = self.loss(y_tensor, preds)
            print(f"LOSS: {loss}")
            epochs+=1
        
if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn import preprocessing

    X, y = make_classification(
    # same as the previous section
    n_samples=1000, n_features=10, n_informative=9, n_classes=2, n_redundant=0, n_repeated=1, 
    # flip_y - high value to add more noise
    flip_y=0.1, 
    # class_sep - low value to reduce space between classes
    class_sep=0.2
    )

    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
        
    # X = preprocessing.normalize(X)
    

    # X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=42)
    X_train = train_images.reshape(train_images.shape[0], -1)
    X_test = test_images.reshape(test_images.shape[0], -1)

    nn = NeuralNet(4, ["sigmoid","sigmoid","softmax"], [300,200,100], 28*28, loss="ce")
    
    nn.train(X_train,train_labels)
    # nn.train(X_train, y_train)
    

