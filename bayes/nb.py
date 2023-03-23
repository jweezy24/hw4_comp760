import numpy as np
import pandas as pd
import sklearn
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB,MultinomialNB

label_map = ["English","Japanese", "Spanish"]

class NaiveBayes:

    def __init__(self,X,y,dist="gaus",alpha=0.5):
        #Sets classes 
        self.classes = np.unique(y)
        print(X.shape)
        
        self.dist = dist
        self.alpha = alpha
        #Do it all
        self.set_priors_and_likelihoods_and_fit(X,y)
        

    def set_priors_and_likelihoods_and_fit(self,X,y):
        #init priors
        self.priors = np.ones(len(self.classes))
        #init likelihoods
        self.likelihoods = []
        

        for i,c in enumerate(self.classes):
            #Gets a list of when y == c, only works for single labeled dataset
            X_c = X[y == c]
            #This sets the prior proabilities for each class in the dataset.
            #We can think about it as finding P(Y)

            if self.dist == "gaus":
                self.priors[i] = X_c.shape[0]/X.shape[0]
            else:
                #Definition of priors in homework

                print(f"Prior Probabilities: {(X_c.shape[0]+self.alpha)}/{(X.shape[0]+(len(self.classes)*self.alpha))} for {label_map[i]} ")

            if self.dist == "gaus":
                means = []
                vars = []
                for j in range(X.shape[1]):
                    #Our likelihood parameters are the mean and std of the distribution of each feature in each class.
                    means.append( X_c[:,j].mean())
                    vars.append(X_c[:,j].var())
                
                self.likelihoods.append((np.array(means),np.array(vars)))
            
        if self.dist == "hw4":
            self.generate_likelihoods_hw4(X,y)
        
    def predict(self,X):
        preds = []
        for p in X:
            posteriors = []
            #Make column vector
            # p = p.reshape(len(p),-1)
            for i,c in enumerate(self.classes):
                # grab prior for ith class
                prior = np.log(self.priors[i])
                # Calculate the posterior
                posterior = np.sum(np.log(self.calculate_likelihood(p, i))) + prior
                # Save Result
                posteriors.append(posterior)
            #Save the class with the highest probability per point.
            preds.append(self.classes[np.argmax(posteriors)])
        
        #Return the class with the highest probability.
        return preds

    def predict_hw4(self,X):
        # Compute the log-likelihood of each class for each document
        log_likelihood = X @ np.log(self.feature_prob.T) + np.log(self.priors)

        # print(log_likelihood)
        # Choose the class with the highest log-likelihood
        if len(log_likelihood.shape) > 1:
            return np.argmax(log_likelihood, axis=1)
        else:
            return np.argmax(log_likelihood)
    
    def calculate_likelihood(self, X, idx):
  
        
        #Get mean and std of class from earlier
        mean, std = self.likelihoods[idx]
        #Threshold for the std
        a = np.exp( - ((X - mean)**2 /(4*std) ))
        b = (1 / (np.sqrt(2 * np.pi * std)))

        return a*b
    

    def generate_likelihoods_hw4(self,X,y):
        self.feature_count = np.zeros((len(self.classes), 27))
        for i in range(len(self.classes)):
            X_i = X[y == i]
            self.feature_count[i,:] = np.sum(X_i, axis=0)
        
        self.feature_count = self.feature_count + self.alpha
        self.feature_prob = self.feature_count / np.sum(self.feature_count, axis=1, keepdims=True)
        
        if False:
            string = ""
            i = 0
            for row in self.feature_prob:
                c = 0
                string+= f"Feature probabilities for {label_map[i]}\n"
                for ele in row:
                    if c == 26:
                        character = "' '"
                    else:
                        character = chr(ord("a")+c) 
                    
                    string+= f" {character}:{ele} "
                    c+=1
                string+= "\n"
                i+=1
            print(string)

if __name__ == "__main__":
    
    # X,y = make_classification(1000,n_classes=5,n_informative=5,n_features=5,n_redundant=0)
    from parse_dataset import *
    X,y = load_dataset()
    
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    X_train,y_train = X["training"],np.array(y["training"])
    X_test,y_test = X["testing"],np.array(y["testing"]) 
    
    nb_model = NaiveBayes(X_train,y_train,dist="hw4")
    
    # model = GaussianNB()
    model = MultinomialNB(alpha=0.5)
    model.fit(X_train,y_train)
    predicted = model.predict(X_test)


    error = 0
    error2 = 0

    e_10 = [164,32,53,57,311,55,51,140,140,3,6,85,64,139,182,53,3,141,186,225,65,31,47,4,38,2,520]

    confusion_matrix = np.zeros((3,3))

    ps = nb_model.predict_hw4(X_test)
    for i in range(X_test.shape[0]):
        tmp = ps[i]
        tmp2 = predicted[i]
        if tmp != y_test[i]:
            error+=1
        if tmp2 != y_test[i]:
            error2 +=1
        confusion_matrix[tmp,y_test[i]] += 1

    error = error/X_test.shape[0]
    error2 = error2/X_test.shape[0]
    print(f"My Model Accuracy = {1-error}\tSklearn's Model Accuracy = {1-error2}")
    print(f"Confusion matrix of choices = {confusion_matrix}")

    nb_model.predict_hw4(e_10)

