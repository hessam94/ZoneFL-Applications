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

f = open("metric.txt", "a")
f2 = open("TestSequence.csv", "a")
#dataset_test = loadtxt('TestSequence.csv', delimiter=',')
pd.read_csv('data/Activity/TestSequence.csv', encoding='utf-8')
zonelist =[4,8,9,12,13,14,17,18,19]
#dataset_test=np.array( [x for x in dataset_test if x[6] == 19])
dataset_test = dataset_test.values
X_train = user_data[:, 0:3]
X_train = np.asarray(X_train).astype(np.float32)
x_test = dataset_test[:,1:6]
y_test = dataset_test[:,7]
x_time = dataset_test[:,1]
x_carid = dataset_test[:,0]

model = load_model('activity.h5')
predictions=model.predict(x_test)   
test_model(x_test, y_test, model,0 , f)
for i in range(len(x_test)):
	print('%d,%d,%d,%d' % (x_carid[i],x_time[i], predictions[i], y_test[i]) , file= f2)  


f.close()
f2.close()
