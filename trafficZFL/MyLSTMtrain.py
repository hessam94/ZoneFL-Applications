"""
Train the NN model.
"""
import sys
import warnings
import argparse
import keras
import numpy as np
import pandas as pd
import tensorflow.keras.metrics  as mtr
from keras.models import load_model
from data import process_data
from model import Mymodel
from keras.models import Model
from keras.callbacks import EarlyStopping
import tensorflow as tf
warnings.filterwarnings("ignore")
from numpy import loadtxt
from FLutils import *

def train_model(model, X_train, y_train, name, config,X_test,Y_test):
    f2 = open("result8.csv", "a")
    """train
    train a single model.

    # Arguments
        model: Model, NN model to train.
        X_train: ndarray(number, lags), Input data for train.
        y_train: ndarray(number, ), result data for train.
        name: String, name of model.
        config: Dict, parameter for train.
    """

    #model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=['sparse_categorical_accuracy'])
    # early = EarlyStopping(monitor='val_loss', patience=30, verbose=0,
    # mode='auto')
    hist = model.fit(X_train, y_train,
        batch_size=config["batch"],
        epochs=config["epochs"],
        validation_split=0.05)

    scores2 = model.evaluate(X_train, y_train)
    print('Accuracy on test data: {}% \n Error on test data: {}'.format(scores2[1], 1 - scores2[1]))   
    
    scores1 = model.evaluate(X_test, Y_test)
    print('Accuracy on test data: {}% \n Error on test data: {}'.format(scores1[1], 1 - scores1[1]))  
    predictions=model.predict(X_test)
    for i in range(len(X_test)):
        print('%d,%d' % ((np.argmax(predictions[i])), Y_test[i]),file=f2 )
    model.save('model/' + name + '.h5')
    #df = pd.DataFrame.from_dict(hist.history)
    #df.to_csv('model/' + name + ' loss.csv', encoding='utf-8', index=False)

def train_seas(models, X_train, y_train, name, config):
    """train
    train the SAEs model.

    # Arguments
        models: List, list of SAE model.
        X_train: ndarray(number, lags), Input data for train.
        y_train: ndarray(number, ), result data for train.
        name: String, name of model.
        config: Dict, parameter for train.
    """

    temp = X_train
    # early = EarlyStopping(monitor='val_loss', patience=30, verbose=0,
    # mode='auto')

    for i in range(len(models) - 1):
        if i > 0:
            p = models[i - 1]
            hidden_layer_model = Model(input=p.input,
                                       output=p.get_layer('hidden').output)
            temp = hidden_layer_model.predict(temp)

        m = models[i]
        m.compile(loss="mse", optimizer="rmsprop", metrics=['mape'])

        m.fit(temp, y_train, batch_size=config["batch"],
              epochs=config["epochs"],
              validation_split=0.05)

        models[i] = m

    saes = models[-1]
    for i in range(len(models) - 1):
        weights = models[i].get_layer('hidden').get_weights()
        saes.get_layer('hidden%d' % (i + 1)).set_weights(weights)

    train_model(saes, X_train, y_train, name, config,X_test,y_test)


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",
        default="gru",
        help="Model to train.")
    args = parser.parse_args()

    config = {"batch": 64, "epochs": 10}

    lags = 3
    seqLen=8 -1;
    f2 = open("result/Z34.csv", "a")
    #dataset_train = loadtxt('data/Activity/ActivityTrain.csv', delimiter=',')
    dataset_train = pd.read_csv('data/Activity/TrainSeq8.csv', encoding='utf-8')
    dataset_train = dataset_train.values
    dataset_train=np.array( [x for x in dataset_train if x[8] == 34])

    dataset_test = pd.read_csv('data/Activity/TestSeq8.csv', encoding='utf-8')
    dataset_test = dataset_test.values
    dataset_test=np.array( [x for x in dataset_test if x[8] == 34])

    train, test = [], []
    train = np.array(dataset_train)
    np.random.shuffle(train)

    optimizer = keras.optimizers.Adam(lr=0.01)
    #global_model = load_model('model/PreModel.h5')
    global_model = Mymodel.get_lstm([seqLen, 64, 64, 8]) # 14 categories we have , array start from 0 but never can predict zero class
    #global_model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer, metrics=['sparse_categorical_accuracy'])
    global_model.compile(loss="sparse_categorical_crossentropy", optimizer='adam', metrics=[mtr.SparseTopKCategoricalAccuracy(k=3)])
    global_model.save("model/lstmjava.h5")
    X_train = train[:, 0:seqLen]
    X_train = np.asarray(X_train).astype(np.float32)
    Y_train = train[:, seqLen]    
    #y = np.random.randint(0,29,(100,)) #(100,)
    Y_train = np.asarray(Y_train).astype(np.float32)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    test = np.array(dataset_test)
    X_test = test[:, 0:seqLen]
    X_test = np.asarray(X_test).astype(np.float32)
    Y_test = test[:, seqLen]    
    #y = np.random.randint(0,29,(100,)) #(100,)
    Y_test = np.asarray(Y_test).astype(np.float32)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    #train_model(global_model,X_train,Y_train,"PreModel2",config,X_test,Y_test)
    
    comms_round = 2
    userCount = 1
    # total data 144315
    trainSizePerRound = X_train.shape[0]//comms_round
    userDataSize = trainSizePerRound//userCount # each round the data per round is divided to 10 users
    clients_batched = dict()
    


    global_model.summary()

    for comm_round in range(comms_round):
            print("round_%d" %( comm_round))
            scaled_local_weight_list = list()
            global_weights = global_model.get_weights()
            np.random.shuffle(train) 
            temp_data = train[:]
            #temp_data = train[comm_round * trainSizePerRound : trainSizePerRound*(comm_round +1)]
            #client_idsAndDate =  client_idsAndDate[(comm_round + 1)* 40:]
            #size = len(temp_data)
            #if size == 0:
                #break
            # each user 144 data, 10 user per round
            
            for user in range(userCount):
                user_data = temp_data[user * userDataSize: (user+1)*userDataSize]

                
                X_train = user_data[:, 0:seqLen]
                X_train = np.asarray(X_train).astype(np.float32)
                Y_train = user_data[:, seqLen]    
                #y = np.random.randint(0,29,(100,)) #(100,)
                Y_train = np.asarray(Y_train).astype(np.float32)
                local_model = Mymodel.get_lstm([seqLen, 64, 64, 8])
                X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
                                                         
                local_model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer, metrics=[mtr.SparseTopKCategoricalAccuracy(k=1)])
                local_model.set_weights(global_weights)
                
                #local_model.fit(X_train, Y_train, batch_size=config["batch"],
                                            #epochs=config["epochs"])

                local_model.fit(X_train, Y_train,epochs=config["epochs"])
                predictions=local_model.predict(X_test)
                #scaling_factor = 1 / len(10)
                scaling_factor = 1 / userCount
                scaled_weights = scale_model_weights(local_model.get_weights(), scaling_factor)
                scaled_local_weight_list.append(scaled_weights)
                K.clear_session()
                #predictions=local_model.predict(X_test)
                #for i in range(len(X_test)):
                       #print('%d,%d' % ((np.argmax(predictions[i])), Y_test[i]) )
                #train_model(m, X_train, Y_train,"testLSTM", config)

            
            average_weights = sum_scaled_weights(scaled_local_weight_list)
            global_model.set_weights(average_weights)
            
    
    
    #global_model.save("activity.h5")
    scores2 = global_model.evaluate(X_train, Y_train)
    print('Accuracy on train data: {}% \n Error on test data: {}'.format(scores2[1], 1 - scores2[1]))   
    
    scores1 = global_model.evaluate(X_test, Y_test)
    print('Accuracy on test data: {}% \n Error on test data: {}'.format(scores1[1], 1 - scores1[1]))  

    
    predictions=global_model.predict(X_test)
    for i in range(len(X_test)):
        print('%d,%d' % ((np.argmax(predictions[i])), Y_test[i]),file= f2 )
    f2.close()
    #global_model.save("model/Cent.h5")
if __name__ == '__main__':
    main(sys.argv)

