from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf
from chart import plot_results
from keras.models import load_model
from keras.utils.vis_utils import plot_model
# load the dataset

dataset = loadtxt('data/train.csv', delimiter=',')
dataset_test = loadtxt('data/test.csv', delimiter=',')
f2 = open("predictionNoFL.csv", "a")
# split into input (X) and output (y) variables , below line works for india diabetes dataset

# my regression traffic dataset
X = dataset[:,1:6]
y = dataset[:,7]

x_test = dataset_test[:,1:6]
y_test = dataset_test[:,7]
x_time = dataset_test[:,1]

names = ['deep']
# define the keras model
model = Sequential()
model.add(Dense(8, input_dim=5, activation='relu'))
model.add(Dense(5, activation='relu'))
model.add(Dense(1,activation="linear", kernel_initializer='normal'))

# compile the keras model
loss = ['mean_absolute_error']
model.compile(loss=loss, optimizer='adam',metrics =[tf.keras.metrics.RootMeanSquaredError()])

# fit the keras model on the dataset
model.fit(X, y, epochs=50, batch_size=32)
#model.save(   'reg.h5')
# evaluate the keras model
#_, accuracy = model.evaluate(X, y)
loss,acc = model.evaluate(x_test,y_test)
#print('Accuracy: %.2f' % (accuracy*100))

#model = load_model('reg.h5')


# make class predictions with the model  predict_classes(X) for classification
y_preds = []
predictions = model.predict(x_test)

file =  'chart.png'
#plot_model(model, to_file=file, show_shapes=True)
#predictions = scaler.inverse_transform(predicted.reshape(-1, 1)).reshape(1, -1)[0]
y_preds.append(predictions[:])
#y_test = y;
#plot_results(y_test[:], y_preds, names)

# summarize the first 5 cases
for i in range(len(x_test)):
	print('%d,%d,%d' % (x_time[i], predictions[i], y_test[i]), file= f2)
