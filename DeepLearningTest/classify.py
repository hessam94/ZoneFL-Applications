from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf
# load the dataset
dataset = loadtxt('data/pima-indians-diabetes.csv', delimiter=',')

# split into input (X) and output (y) variables , below line works for india diabetes dataset
X = dataset[:,0:8]
y = dataset[:,8]



# define the keras model
model = Sequential()

model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


# compile the keras model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# fit the keras model on the dataset
model.fit(X, y, epochs=1, batch_size=10)

# evaluate the keras model
print ("now evaluate\n")
_, accuracy = model.evaluate(X, y)
print('Accuracy: %.2f' % (accuracy*100))

# make class predictions with the model  predict_classes(X) for classification
predictions = model.predict_classes(X)

# summarize the first 5 cases
for i in range(5):
	print('%s => %d (expected %d)' % (X[i].tolist(), predictions[i], y[i]))
