from sklearn import metrics
from sklearn.metrics import classification_report,confusion_matrix
import h5py
import keras
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv1D, MaxPooling1D, ZeroPadding1D,AveragePooling1D,regularizers
from keras import backend as K
from keras.utils import np_utils
from keras.layers.normalization import BatchNormalization
from scipy.stats import mode
#import theano 

import matplotlib.pyplot as plt
import numpy as np
from keras import backend as K
#K.set_image_dim_ordering('th')
from random import shuffle
from keras.callbacks import ModelCheckpoint
import os
from numpy import *
# SKLEARN
from sklearn.utils import shuffle
#from sklearn.cross_validation import train_test_split

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
from keras.layers import LSTM
import scipy.io
import pickle
import os
from keras.models import Model
from keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize


from scipy.stats import mode
#import theano 
import matplotlib.pyplot as plt
import numpy as np

#K.set_image_dim_ordering('th')
from random import shuffle

import os
from numpy import *
# SKLEARN
from sklearn.utils import shuffle

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import random 
from random import randint
from sklearn.svm import SVC
#%%

def list_gen(lst):
	if len(lst)>100:
		lst_new=lst[0:100]
	elif len(lst)<100:
		diff=100-len(lst)
		low_limit=len(lst)-diff
		lst_new=np.hstack(lst,np.arange(low_limi,len(lst),1,dtype=int))
	else:
		lst_new=lst
	return	lst_new



#%%

param_G=np.load("~/sound8.npy", encoding = 'latin1',allow_pickle=True).item()
initia_weights1=[np.reshape(param_G['conv1']['weights'],(64,1,16)),param_G['conv1']['biases'],param_G['conv1']['gamma'],param_G['conv1']['beta'],param_G['conv1']['mean'],param_G['conv1']['var'],np.reshape(param_G['conv2']['weights'],(32,16,32)),param_G['conv2']['biases'],param_G['conv2']['gamma'],param_G['conv2']['beta'],param_G['conv2']['mean'],param_G['conv2']['var']]#,np.reshape(param_G['conv3']['weights'],(16,32,64)),param_G['conv3']['biases'],param_G['conv3']['gamma'],param_G['conv3']['beta'],param_G['conv3']['mean'],param_G['conv3']['var']]#,np.reshape(param_G['conv4']['weights'],(8,64,128)),param_G['conv4']['biases'],param_G['conv4']['gamma'],param_G['conv4']['beta'],param_G['conv4']['mean'],param_G['conv4']['var']]#,np.reshape(param_G['conv5']['weights'],(4,128,256)),param_G['conv5']['biases'],param_G['conv5']['gamma'],param_G['conv5']['beta'],param_G['conv5']['mean'],param_G['conv5']['var'],np.reshape(param_G['conv6']['weights'],(4,256,512)),param_G['conv6']['biases'],param_G['conv6']['gamma'],param_G['conv6']['beta'],param_G['conv6']['mean'],param_G['conv6']['var'],np.reshape(param_G['conv7']['weights'],(4,512,1024)),param_G['conv7']['biases'],param_G['conv7']['gamma'],param_G['conv7']['beta'],param_G['conv7']['mean'],param_G['conv7']['var']]#,np.reshape(param_G['conv8']['weights'],(8,1024,1000)),param_G['conv8']['biases'],np.reshape(param_G['conv8_2']['weights'],(8,1024,401)),param_G['conv8_2']['biases']]


#%% note that 0 is normal and 1 is abnormal in log mel features.....TNR is the measure of abnormals are classififed as abnormal...

model =Sequential()

model.add(Conv1D(16,64,strides=2,input_shape=(2000,1))) #layer1
model.add(ZeroPadding1D(padding=16))
model.add(BatchNormalization()) #layer2
convout1= Activation('relu')
model.add(convout1) #layer3



model.add(MaxPooling1D(pool_size=8, padding='valid')) #layer4
#
#
model.add(Conv1D(32,32,strides=2)) #layer5
model.add(ZeroPadding1D(padding=8))
model.add(BatchNormalization()) #layer6
convout2= Activation('relu')
model.add(convout2) #layer7
#model.add(Dropout(0.5))
model.set_weights(initia_weights1)

#%%
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])

model.summary()
model_D4=model
#%%


os.chdir('Path of raw audio segments saved in numpy format')   #Path of the raw data sampled at 8kHz anf od 2000 dimension


normal_data_6dB=np.average(model_D4.predict(normal_data_6dB),1)
abnormal_data_6dB=np.average(model_D4.predict(abnormal_data_6dB),1)

np.save('normal_data_case4_SoundNet',normal_data_6dB)
np.save('abnormal_data_case4_SoundNet',abnormal_data_6dB)
