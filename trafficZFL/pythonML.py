import tensorflow as tf
import h5py
print (tf.__version__)


# To read the HDF file
f = h5py.File('TrafficFlowPrediction/model/gru.h5', 'r')

print("Layers: %s" % f.keys())
# Get dataset of a particular layer, for example <IMG_MIR_TEMP>
data = f['model_weights']

print ("hiiii")