import numpy as np
import random
#import cv2
import os
#from imutils import paths
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
from keras.utils.vis_utils import plot_model
from chart import plot_results

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import backend as K
from numpy import loadtxt

from FLutils import *
zonelist =[4,8,9,12,13,14,17,18,19]
#dataset = loadtxt('data/tiny.csv', delimiter=',')
dataset_train = loadtxt('data/train.csv', delimiter=',')
dataset_validation = loadtxt('data/validation.csv', delimiter=',')
dataset_test = loadtxt('data/test.csv', delimiter=',')
#dataset_train=np.array([x for x in dataset_train if x[6] == 19])
#dataset_test=np.array( [x for x in dataset_test if x[6] == 19])
#dataset_train = dataset_test
client_ids = list(set( dataset_train[:,0])) # make it call by value , if you use this then it is call by reference --> client_ids = dataset[:,1]
#x_validation = dataset_validation[:,2:6]
#y_validation = dataset_validation[:,6]
x_test = dataset_test[:,1:6]
x_zones = dataset_train[:,6]
y_test = dataset_test[:,7]
x_time = dataset_test[:,1]
x_carid = dataset_test[:,0]

    
#process and batch the test set  
#test_batched = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(len(y_test))

comms_round = 5
    
loss = ['mean_absolute_error']
metrics = ['mean_squared_error']

#initialize global model

#model = Sequential()
#model.add(Dense(units=8, activation='relu', input_dim=5))
#model.add(Dense(units=1, activation='softmax'))
#model.compile(loss=loss, optimizer='adam', metrics=metrics)

#model.save('simple_mlp2.h5')


smlp_global = SimpleMLP()
global_model = smlp_global.build()    
global_model.compile(loss=loss, optimizer='adam',metrics =[tf.keras.metrics.RootMeanSquaredError()])
#global_model.save('trafficGlobal.h5')


#model2 = Sequential()
#model2.add(Dense(8, input_dim=5, activation='relu'))
#model2.add(Dense(5, activation='relu'))
#model2.add(Dense(1,activation="linear", kernel_initializer='normal'))
#model2.compile(loss=loss, optimizer='adam',metrics =[tf.keras.metrics.RootMeanSquaredError()]);
#model2.save('trafficGlobal.h5');
#commence global training loop

f = open("metric.txt", "a")
#f2 = open("prediction19.csv", "a")
iteration = []
metricResults=[]
for comm_round in range(comms_round):
    
    temp_ids = create_clientsByID(client_ids,40)
    size = len(temp_ids)
    if size==0:
        break


    client_ids = list(set(client_ids) - set (temp_ids) )

    # get the global model's weights - will serve as the initial weights for all local models
    global_weights = global_model.get_weights()
    
    #initial list to collect local model weights after scalling
    scaled_local_weight_list = list()
    
    #loop through each client and create new local model

    clients_batched = dict()
    
    for id in temp_ids:
        tempclient_list=np.array( [x for x in dataset_train if x[0] == id])
        x_train =tempclient_list [:,1:6]
        y_train =tempclient_list [:,7]
        
        dd=tf.data.Dataset.from_tensor_slices((list(x_train), list(y_train))).batch(32)
        clients_batched[id] =dd

    for id in temp_ids:
        smlp_local = SimpleMLP()
        local_model = smlp_local.build()
        local_model.compile(loss=loss, optimizer='adam',metrics = tf.keras.metrics.RootMeanSquaredError())
        
        #set local model weight to the weight of the global model
        local_model.set_weights(global_weights)
        
        #fit local model with client's data
        local_model.fit(clients_batched[id], epochs=10,verbose =0)
        #local_model.fit(x_train,y_train, epochs=25, verbose=0)
        
        #scale the model weights and add to list
        #scaling_factor = weight_scalling_factor(clients_batched, id)
        scaling_factor = 1/len(temp_ids)
        scaled_weights = scale_model_weights(local_model.get_weights(), scaling_factor)
        scaled_local_weight_list.append(scaled_weights)
        
        #clear session to free memory after each communication round
        K.clear_session()
        
    #to get the average over all the local model, we simply take the sum of the scaled weights
    average_weights = sum_scaled_weights(scaled_local_weight_list)
    
    #update global model 
    global_model.set_weights(average_weights)

        #test global model and print out metrics after each communications round
    #for(x_test, y_test) in test_batched:
    print('comm_round: {}'.format(comm_round))
    global_acc, global_loss = test_model(x_test, y_test, global_model, comm_round , f)
    metricResults.append(global_acc)
    iteration.append(comm_round)

predictions=global_model.predict(x_test)   

#plot_results(iteration, metricResults,names)
test_model(x_test, y_test, global_model, comm_round , f)
#global_model.save('zone19.h5')
#for i in range(len(x_test)):
    #print('%d,%d,%d,%d' % (x_carid[i],x_time[i], predictions[i], y_test[i]) )   

#f.close()
#f2.close()

