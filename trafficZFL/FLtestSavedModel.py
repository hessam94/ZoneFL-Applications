import numpy as np
import random
#import cv2
import os
#from imutils import paths
from sklearn.model_selection import train_test_split

from numpy import loadtxt
from keras.models import load_model
import numpy as np
import pandas as pd
from FLutils import *

#f = open("metric.txt", "a")
#f2 = open("result/C18.csv", "a")
#dataset_test = loadtxt('TestSequence.csv', delimiter=',')
dataset_test = pd.read_csv('data/Activity/TestSeq8.csv', encoding='utf-8')
zonelist = [26,21,33,18,24,25,32] # activity app zone areas
#zonelist =[4,8,9,12,13,14,17,18,19] # traffic app zone area
seqLen =8 -1
dataset_test = dataset_test.values

    
model = load_model('model/CentralActivityk1.h5')
for z in zonelist:
    temp_test=np.array( [x for x in dataset_test if x[8] == z])
    test = np.array(temp_test)
    X_test = test[:, 0:seqLen]
    X_test = np.asarray(X_test).astype(np.float32)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    Y_test = test[:, seqLen]   
    Y_test = np.asarray(Y_test).astype(np.float32)

    
    predictions=model.predict(X_test)
    scores2 = model.evaluate(X_test, Y_test)
    print('Accuracy on test data for zone {}: {}% \n Error on test data: {}  \n'.format(z,scores2[1], 1 - scores2[1]))    
#for i in range(len(X_test)):
	#print('%d,%d' % ((np.argmax(predictions[i])), Y_test[i]) , file= f2)  


#f.close()
#f2.close()
