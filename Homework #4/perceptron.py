#-------------------------------------------------------------------------
# AUTHOR: Noah Reef
# FILENAME: perceptron.py
# SPECIFICATION: MLP and Perceptron classifier for hand written digits
# FOR: CS 4210- Assignment #4
# TIME SPENT: 30 minutes
#-----------------------------------------------------------*/

#IMPORTANT NOTE: YOU HAVE TO WORK WITH THE PYTHON LIBRARIES numpy AND pandas to complete this code.
#importing some Python libraries
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier #pip install scikit- learn==0.18.rc2 if needed
import numpy as np
import pandas as pd

n = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
r = [True, False]

df = pd.read_csv('Homework #4/optdigits.tra', sep=',', header=None) #reading the data by using Pandas library

X_training = np.array(df.values)[:,:64] #getting the first 64 fields to form the feature data for training
y_training = np.array(df.values)[:,-1] #getting the last field to form the class label for training

df = pd.read_csv('Homework #4/optdigits.tes', sep=',', header=None) #reading the data by using Pandas library

X_test = np.array(df.values)[:,:64] #getting the first 64 fields to form the feature data for test
y_test = np.array(df.values)[:,-1] #getting the last field to form the class label for test

mlp_accuracy = 0
mlp_rate = 0
mlp_shuffle = False

percep_accuracy = 0
percep_rate = 0
percep_shuffle = False

for n_k in n: #iterates over n
    for r_k in r: #iterates over r
        #iterates over both algorithms
        models = ["Single","Multi"]
        for model in models: #iterates over the algorithms

            #Create a Neural Network classifier
            clf = Perceptron(eta0=n_k, shuffle=r_k, max_iter=1000)

            if model == "Single":
                clf = Perceptron(eta0=n_k, shuffle=r_k, max_iter=1000)

            else:
                clf = MLPClassifier(activation='logistic', learning_rate_init=n_k, shuffle=r_k, max_iter=1000)

            clf.fit(X_training, y_training)

            acc = 0

            for (x_testSample, y_testSample) in zip(X_test, y_test):
                if clf.predict([x_testSample]) == y_testSample:
                    acc += 1

            acc = acc/len(X_test)

            if model == "Single" and acc > percep_accuracy:
                percep_accuracy = acc
                percep_rate = n_k
                percep_shuffle = r_k
                print("Highest Perception Accuracy so far: " + str(percep_accuracy) + ", Parameters: learning_rate=" + str(percep_rate) + ",shuffle=" + str(percep_shuffle))

            if model == "Multi" and acc > mlp_accuracy:
                mlp_accuracy = acc
                mlp_rate = n_k
                mlp_shuffle = r_k
                print("Highest MLP Accuracy so far: " + str(percep_accuracy) + ", Parameters: learning_rate=" + str(mlp_rate) + ",shuffle=" + str(mlp_shuffle))

#if Perceptron then
# clf = Perceptron() #use those hyperparameters: eta0 = learning rate, shuffle = shuffle_the_training_data, max_iter=1000
#else:
# clf = MLPClassifier() #use those hyperparameters: activation='logistic', learning_rate_init = learning rate, hidden_layer_sizes = number of neurons in the ith hidden layer,
# shuffle = shuffle the training data, max_iter=1000
#-->add your Pyhton code here
#Fit the Neural Network to the training data
#make the classifier prediction for each test sample and start computing its accuracy
#hint: to iterate over two collections simultaneously with zip() Example:
#for (x_testSample, y_testSample) in zip(X_test, y_test):
#to make a prediction do: clf.predict([x_testSample])
#--> add your Python code here
#check if the calculated accuracy is higher than the previously one calculated for each classifier. If so, update the highest accuracy
#and print it together with the network hyperparameters
#Example: "Highest Perceptron accuracy so far: 0.88, Parameters: learning rate=0.01, shuffle=True"
#Example: "Highest MLP accuracy so far: 0.90, Parameters: learning rate=0.02, shuffle=False"
#--> add your Python code here