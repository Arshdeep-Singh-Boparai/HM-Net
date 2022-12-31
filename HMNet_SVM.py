# -*- coding: utf-8 -*-
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 01:48:49 2020

@author: arshdeep
"""
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
from sklearn import metrics
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

#%%

def list_gen(th,lst):
    if len(th) > 100:
        lst_new=np.append(lst[0:100],len(th)-1)
    elif len(th) < 100:
        diff=100-len(th)
        lst_new=np.append(np.tile(0,diff+1),np.arange(0,len(th),1,dtype=int))
    else:
        lst_new=np.append(0,lst)
    return lst_new

    
#%% note that 0 is normal and 1 is abnormal in log mel features.....TNR is the measure of abnormals are classififed as abnormal...
model =Sequential()


#layer1
model.add(Conv1D(16,64,strides=2,input_shape=(2000,1), kernel_regularizer=regularizers.l2(0.2),trainable= False)) 
model.add(ZeroPadding1D(padding=16))
model.add(BatchNormalization(trainable=True)) #layer2
convout1= Activation('relu')
model.add(convout1) #layer3
model.add(Dropout(0.2))
model.add(AveragePooling1D(pool_size=4, padding='valid')) #layer4

#layer 2
model.add(Conv1D(32,32,strides=2,kernel_regularizer=regularizers.l2(0.2),trainable=False)) #layer5
model.add(ZeroPadding1D(padding=8))
model.add(BatchNormalization(trainable=True)) #layer6
convout2= Activation('relu')
model.add(convout2) #layer7
model.add(Dropout(0.2))
model.add(AveragePooling1D(pool_size=4, padding='valid')) #layer8

#layer 3
model.add(Conv1D(64,16,strides=2,kernel_regularizer=regularizers.l2(0.2),trainable=False)) #layer9
model.add(ZeroPadding1D(padding=4))
model.add(BatchNormalization(trainable=True)) #layer10
convout3= Activation('relu')
model.add(convout3) #layer11
model.add(Dropout(0.2))
model.add(AveragePooling1D(pool_size=4, padding='valid'))


model.add(Flatten())
model.add((Dense(64,trainable=True)))
model.add((Activation('relu')))
model.add(Dropout(0.2))


model.add(Dense(2))
model.add(Activation('softmax'))
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(lr=0.01,decay=0.001),
              metrics=['accuracy'])

model.load_weights('~/HMNet_supervised_all10.hdf5') #pre-trained HM-Net weights

model.summary()

#%% Data prepration/load /

from random import randint
from sklearn.svm import SVC

os.chdir('~/Slider/ID_06')  #Path of the raw data sampled at 8kHz anf od 2000 dimension
normal_data_6dB=np.load('normal_data_6dB.npy').astype('float32')  # raw audio data saved for each 250ms segments from 6dB (MIMII) normal samples
normal_data_0dB=np.load('normal_data_0dB.npy').astype('float32')# raw audio data saved for each 250ms segments from 6dB (MIMII) normal samples
normal_data_C6dB=np.load('normal_data_-6dB.npy').astype('float32')
abnormal_data_6dB=np.load('abnormal_data_6dB.npy').astype('float32')  # raw audio data saved for each 250ms segments from 6dB (MIMII) abnormal samples
abnormal_data_0dB=np.load('abnormal_data_0dB.npy').astype('float32')# raw audio data saved for each 250ms segments from 6dB (MIMII) abnoraml samples
abnormal_data_C6dB=np.load('abnormal_data_-6dB.npy').astype('float32')
labels_test=np.load('labels_test.npy').astype('float32')


normal_data_6dB=np.reshape(normal_data_6dB, (np.shape(normal_data_6dB)[0],np.shape(normal_data_6dB)[1],1))
abnormal_data_6dB=np.reshape(abnormal_data_6dB, (np.shape(abnormal_data_6dB)[0],np.shape(abnormal_data_6dB)[1],1))


normal_data_0dB=np.reshape(normal_data_0dB, (np.shape(normal_data_0dB)[0],np.shape(normal_data_0dB)[1],1))
abnormal_data_0dB=np.reshape(abnormal_data_0dB, (np.shape(abnormal_data_0dB)[0],np.shape(abnormal_data_0dB)[1],1))

normal_data_C6dB=np.reshape(normal_data_C6dB, (np.shape(normal_data_C6dB)[0],np.shape(normal_data_C6dB)[1],1))
abnormal_data_C6dB=np.reshape(abnormal_data_C6dB, (np.shape(abnormal_data_C6dB)[0],np.shape(abnormal_data_C6dB)[1],1))


#%% feature extraction from HM-Net 

model_D4=Model(inputs=model.input, outputs=model.get_layer('average_pooling1d_3').output)

normal_data_6dB=np.average(model_D4.predict(normal_data_6dB),1)
abnormal_data_6dB=np.average(model_D4.predict(abnormal_data_6dB),1)

normal_data_0dB=np.average(model_D4.predict(normal_data_0dB),1)
abnormal_data_0dB=np.average(model_D4.predict(abnormal_data_0dB),1)

normal_data_C6dB=np.average(model_D4.predict(normal_data_C6dB),1)
abnormal_data_C6dB=np.average(model_D4.predict(abnormal_data_C6dB),1)

print(np.shape(normal_data_6dB),np.shape(abnormal_data_6dB))
print(np.shape(normal_data_0dB),np.shape(abnormal_data_0dB))
print(np.shape(normal_data_C6dB),np.shape(abnormal_data_C6dB))



os.chdir('/home/arshdeep/Arsh_industry_analsyis/Supervised_SVM_HMNet_35trails/AUC_ROC_CURVES/HMNet/Slider/ID_06')

#%% SVM classification for 35-reandom trails when different number of trainign examples are used for SVM training


loop_iter=35


#%%
Num_examples=1 # here 1 sound example means: all 40 segments are used for training from the 1 examples
TPR=[]
FPR=[]
TNR=[]
FNR=[]
AUC=[]
th_p=[]
th_n=[]
auc_iter=0
print('T=1...RUNNING')

for i in range(loop_iter):
    T=Num_examples
    list_abn=[]
    random.seed(i)
    for i in range(T):
        v=randint(0,int(len(abnormal_data_6dB)/40)-1)
        a=list(range(40*v,40*v+40))
        list_abn.append(a)
    list_norm=[]
    random.seed(i)
    for i in range(T):
        v=randint(0,int(len(normal_data_6dB)/40)-1)
        a=list(range(40*v,40*v+40))
        list_norm.append(a)
    list_abn_train=np.hstack(list_abn)
    list_norm_train=np.hstack(list_norm)
    list_abn_test=list(set(list(range(0,int(len(abnormal_data_6dB)))))-set(list_abn_train))
    list_norm_test=list(set(list(range(0,int(len(normal_data_6dB)))))-set(list_norm_train))
    x_train_abnormal=np.vstack((abnormal_data_6dB[list_abn_train,:],abnormal_data_6dB[list_abn_train,:],abnormal_data_C6dB[list_abn_train,:]))  #100 abnormal
    x_train_normal=np.vstack((normal_data_6dB[list_norm_train,:],normal_data_0dB[list_norm_train,:],normal_data_C6dB[list_norm_train,:])) # 300 normal 
    x_train_06=np.vstack((x_train_abnormal,x_train_normal))
    label_train_06= np.hstack((np.tile(1,len(x_train_abnormal)),np.tile(0,len(x_train_normal))))
    x_test_abnormal=np.vstack((abnormal_data_6dB[list_abn_test,:],abnormal_data_0dB[list_abn_test,:],abnormal_data_C6dB[list_abn_test,:]))
    x_test_normal=np.vstack((normal_data_6dB[list_norm_test,:],normal_data_0dB[list_norm_test,:],normal_data_C6dB[list_norm_test,:]))
    x_test_06=np.vstack((x_test_abnormal,x_test_normal))
    label_test_06=np.hstack((np.tile(1,len(x_test_abnormal)),np.tile(0,len(x_test_normal))))
    X_train = x_train_06.astype('float32')
    X_test=x_test_06.astype('float32')
    label_test=label_test_06
    svm_train= X_train
    svm_test=X_test
    clf = SVC(kernel='linear', probability=True)
    clf.fit(svm_train, label_train_06)
    pred_label=clf.predict_proba(svm_test)

    score=pred_label
    t=0;
    split_fac=40
    num_segment=split_fac;
    pred_prob=[]
    for i in range(int(np.shape(X_test)[0]/split_fac)):
        pred=np.sum(score[t:t+num_segment,:],0)
        pred_prob.append(pred)
        t=t+num_segment
    pred_prob1=(np.asarray(pred_prob))/40
    y_class= pred_prob1.argmax(axis=-1)
    actual_labels=(label_test[0:np.shape(X_test)[0]:split_fac])
    auc=metrics.roc_auc_score(actual_labels,y_class)
    fpr, tpr, thresholds_p = metrics.roc_curve(actual_labels, pred_prob1[:,1], pos_label=1) #1 means abnormal
    fnr, tnr, thresholds_n = metrics.roc_curve(actual_labels, pred_prob1[:,0], pos_label=0)
    lst_p=np.arange(0,len(thresholds_p),len(thresholds_p)/100,dtype=int)
    lst_n=np.arange(0,len(thresholds_n),len(thresholds_n)/100,dtype=int)    
    lst_new_p=list_gen(thresholds_p,lst_p)
#    lst_new1_p=np.append(list_new_p,np.arange(len(thresholds_p)-10,len(thresholds_p),1,dtype=int))  
    lst_new_n=list_gen(thresholds_n,lst_n)
#    lst_new1_n=np.append(list_new_n,np.arange(len(thresholds_n)-10,len(thresholds_n),1,dtype=int))      
    AUC.append(auc)
    auc_iter=auc_iter+1
    print(auc,'  At trial ', auc_iter)    
    TPR.append(tpr[lst_new_p])
    FPR.append(fpr[lst_new_p])
    FNR.append(fnr[lst_new_n])
    TNR.append(tnr[lst_new_n])
    th_p.append(thresholds_p[lst_new_p])
    th_n.append(thresholds_n[lst_new_n])


TPR_all=np.average(TPR,0)
FPR_all= np.average(FPR,0)
TNR_all=np.average(TNR,0)
FNR_all=np.average(FNR,0)
Th_n=np.average(th_n,0)
Th_p=np.average(th_p,0)

np.save('1/TPR',TPR_all)
np.save('1/FPR',FPR_all)
np.save('1/TNR',TNR_all)
np.save('1/FNR',FNR_all)
np.save('1/AUC',AUC)
np.save('1/Th_p',Th_p)

print('T=1...completed and saved')

#%%


Num_examples=3
TPR=[]
FPR=[]
TNR=[]
FNR=[]
AUC=[]
th_p=[]
th_n=[]
auc_iter=0

for i in range(loop_iter):
    T=Num_examples
    list_abn=[]
    random.seed(i)
    for i in range(T):
        v=randint(0,int(len(abnormal_data_6dB)/40)-1)
        a=list(range(40*v,40*v+40))
        list_abn.append(a)
    list_norm=[]
    random.seed(i)
    for i in range(T):
        v=randint(0,int(len(normal_data_6dB)/40)-1)
        a=list(range(40*v,40*v+40))
        list_norm.append(a)
    list_abn_train=np.hstack(list_abn)
    list_norm_train=np.hstack(list_norm)
    list_abn_test=list(set(list(range(0,int(len(abnormal_data_6dB)))))-set(list_abn_train))
    list_norm_test=list(set(list(range(0,int(len(normal_data_6dB)))))-set(list_norm_train))
    x_train_abnormal=np.vstack((abnormal_data_6dB[list_abn_train,:],abnormal_data_6dB[list_abn_train,:],abnormal_data_C6dB[list_abn_train,:]))  #100 abnormal
    x_train_normal=np.vstack((normal_data_6dB[list_norm_train,:],normal_data_0dB[list_norm_train,:],normal_data_C6dB[list_norm_train,:])) # 300 normal 
    x_train_06=np.vstack((x_train_abnormal,x_train_normal))
    label_train_06= np.hstack((np.tile(1,len(x_train_abnormal)),np.tile(0,len(x_train_normal))))
    x_test_abnormal=np.vstack((abnormal_data_6dB[list_abn_test,:],abnormal_data_0dB[list_abn_test,:],abnormal_data_C6dB[list_abn_test,:]))
    x_test_normal=np.vstack((normal_data_6dB[list_norm_test,:],normal_data_0dB[list_norm_test,:],normal_data_C6dB[list_norm_test,:]))
    x_test_06=np.vstack((x_test_abnormal,x_test_normal))
    label_test_06=np.hstack((np.tile(1,len(x_test_abnormal)),np.tile(0,len(x_test_normal))))
    X_train = x_train_06.astype('float32')
    X_test=x_test_06.astype('float32')
    label_test=label_test_06
    svm_train= X_train
    svm_test=X_test
    clf = SVC(kernel='linear', probability=True)
    clf.fit(svm_train, label_train_06)
    pred_label=clf.predict_proba(svm_test)

    score=pred_label
    t=0;
    split_fac=40
    num_segment=split_fac;
    pred_prob=[]
    for i in range(int(np.shape(X_test)[0]/split_fac)):
        pred=np.sum(score[t:t+num_segment,:],0)
        pred_prob.append(pred)
        t=t+num_segment
    pred_prob1=(np.asarray(pred_prob))/40
    y_class= pred_prob1.argmax(axis=-1)
    actual_labels=(label_test[0:np.shape(X_test)[0]:split_fac])
    auc=metrics.roc_auc_score(actual_labels,y_class)
    fpr, tpr, thresholds_p = metrics.roc_curve(actual_labels, pred_prob1[:,1], pos_label=1) #1 means abnormal
    fnr, tnr, thresholds_n = metrics.roc_curve(actual_labels, pred_prob1[:,0], pos_label=0)
    lst_p=np.arange(0,len(thresholds_p),len(thresholds_p)/100,dtype=int)
    lst_n=np.arange(0,len(thresholds_n),len(thresholds_n)/100,dtype=int)    
    lst_new_p=list_gen(thresholds_p,lst_p)
#    lst_new1_p=np.append(list_new_p,np.arange(len(thresholds_p)-10,len(thresholds_p),1,dtype=int))  
    lst_new_n=list_gen(thresholds_n,lst_n)
#    lst_new1_n=np.append(list_new_n,np.arange(len(thresholds_n)-10,len(thresholds_n),1,dtype=int))      
    AUC.append(auc)
    auc_iter=auc_iter+1
    print(auc,'  At trial ', auc_iter)    
    TPR.append(tpr[lst_new_p])
    FPR.append(fpr[lst_new_p])
    FNR.append(fnr[lst_new_n])
    TNR.append(tnr[lst_new_n])
    th_p.append(thresholds_p[lst_new_p])
    th_n.append(thresholds_n[lst_new_n])


TPR_all=np.average(TPR,0)
FPR_all= np.average(FPR,0)
TNR_all=np.average(TNR,0)
FNR_all=np.average(FNR,0)
Th_n=np.average(th_n,0)
Th_p=np.average(th_p,0)

np.save('3/TPR',TPR_all)
np.save('3/FPR',FPR_all)
np.save('3/TNR',TNR_all)
np.save('3/FNR',FNR_all)
np.save('3/AUC',AUC)
np.save('3/Th_p',Th_p)


print('T=3...completed and saved')
#%%

Num_examples=7
TPR=[]
FPR=[]
TNR=[]
FNR=[]
AUC=[]
th_p=[]
th_n=[]
auc_iter=0
for i in range(loop_iter):
    T=Num_examples
    list_abn=[]
    random.seed(i)
    for i in range(T):
        v=randint(0,int(len(abnormal_data_6dB)/40)-1)
        a=list(range(40*v,40*v+40))
        list_abn.append(a)
    list_norm=[]
    random.seed(i)
    for i in range(T):
        v=randint(0,int(len(normal_data_6dB)/40)-1)
        a=list(range(40*v,40*v+40))
        list_norm.append(a)
    list_abn_train=np.hstack(list_abn)
    list_norm_train=np.hstack(list_norm)
    list_abn_test=list(set(list(range(0,int(len(abnormal_data_6dB)))))-set(list_abn_train))
    list_norm_test=list(set(list(range(0,int(len(normal_data_6dB)))))-set(list_norm_train))
    x_train_abnormal=np.vstack((abnormal_data_6dB[list_abn_train,:],abnormal_data_6dB[list_abn_train,:],abnormal_data_C6dB[list_abn_train,:]))  #100 abnormal
    x_train_normal=np.vstack((normal_data_6dB[list_norm_train,:],normal_data_0dB[list_norm_train,:],normal_data_C6dB[list_norm_train,:])) # 300 normal 
    x_train_06=np.vstack((x_train_abnormal,x_train_normal))
    label_train_06= np.hstack((np.tile(1,len(x_train_abnormal)),np.tile(0,len(x_train_normal))))
    x_test_abnormal=np.vstack((abnormal_data_6dB[list_abn_test,:],abnormal_data_0dB[list_abn_test,:],abnormal_data_C6dB[list_abn_test,:]))
    x_test_normal=np.vstack((normal_data_6dB[list_norm_test,:],normal_data_0dB[list_norm_test,:],normal_data_C6dB[list_norm_test,:]))
    x_test_06=np.vstack((x_test_abnormal,x_test_normal))
    label_test_06=np.hstack((np.tile(1,len(x_test_abnormal)),np.tile(0,len(x_test_normal))))
    X_train = x_train_06.astype('float32')
    X_test=x_test_06.astype('float32')
    label_test=label_test_06
    svm_train= X_train
    svm_test=X_test
    clf = SVC(kernel='linear', probability=True)
    clf.fit(svm_train, label_train_06)
    pred_label=clf.predict_proba(svm_test)

    score=pred_label
    t=0;
    split_fac=40
    num_segment=split_fac;
    pred_prob=[]
    for i in range(int(np.shape(X_test)[0]/split_fac)):
        pred=np.sum(score[t:t+num_segment,:],0)
        pred_prob.append(pred)
        t=t+num_segment
    pred_prob1=(np.asarray(pred_prob))/40
    y_class= pred_prob1.argmax(axis=-1)
    actual_labels=(label_test[0:np.shape(X_test)[0]:split_fac])
    auc=metrics.roc_auc_score(actual_labels,y_class)
    fpr, tpr, thresholds_p = metrics.roc_curve(actual_labels, pred_prob1[:,1], pos_label=1) #1 means abnormal
    fnr, tnr, thresholds_n = metrics.roc_curve(actual_labels, pred_prob1[:,0], pos_label=0)
    lst_p=np.arange(0,len(thresholds_p),len(thresholds_p)/100,dtype=int)
    lst_n=np.arange(0,len(thresholds_n),len(thresholds_n)/100,dtype=int)    
    lst_new_p=list_gen(thresholds_p,lst_p)
#    lst_new1_p=np.append(list_new_p,np.arange(len(thresholds_p)-10,len(thresholds_p),1,dtype=int))  
    lst_new_n=list_gen(thresholds_n,lst_n)
#    lst_new1_n=np.append(list_new_n,np.arange(len(thresholds_n)-10,len(thresholds_n),1,dtype=int))      
    AUC.append(auc)
    auc_iter=auc_iter+1
    print(auc,'  At trial ', auc_iter)    
    TPR.append(tpr[lst_new_p])
    FPR.append(fpr[lst_new_p])
    FNR.append(fnr[lst_new_n])
    TNR.append(tnr[lst_new_n])
    th_p.append(thresholds_p[lst_new_p])
    th_n.append(thresholds_n[lst_new_n])


TPR_all=np.average(TPR,0)
FPR_all= np.average(FPR,0)
TNR_all=np.average(TNR,0)
FNR_all=np.average(FNR,0)
Th_n=np.average(th_n,0)
Th_p=np.average(th_p,0)
np.save('7/TPR',TPR_all)
np.save('7/FPR',FPR_all)
np.save('7/TNR',TNR_all)
np.save('7/FNR',FNR_all)
np.save('7/AUC',AUC)
np.save('7/Th_p',Th_p)


print('T=7...completed and saved')

#%%

Num_examples=20
TPR=[]
FPR=[]
TNR=[]
FNR=[]
AUC=[]
th_p=[]
th_n=[]
auc_iter=0


for i in range(loop_iter):
    T=Num_examples
    list_abn=[]
    random.seed(i)
    for i in range(T):
        v=randint(0,int(len(abnormal_data_6dB)/40)-1)
        a=list(range(40*v,40*v+40))
        list_abn.append(a)
    list_norm=[]
    random.seed(i)
    for i in range(T):
        v=randint(0,int(len(normal_data_6dB)/40)-1)
        a=list(range(40*v,40*v+40))
        list_norm.append(a)
    list_abn_train=np.hstack(list_abn)
    list_norm_train=np.hstack(list_norm)
    list_abn_test=list(set(list(range(0,int(len(abnormal_data_6dB)))))-set(list_abn_train))
    list_norm_test=list(set(list(range(0,int(len(normal_data_6dB)))))-set(list_norm_train))
    x_train_abnormal=np.vstack((abnormal_data_6dB[list_abn_train,:],abnormal_data_6dB[list_abn_train,:],abnormal_data_C6dB[list_abn_train,:]))  #100 abnormal
    x_train_normal=np.vstack((normal_data_6dB[list_norm_train,:],normal_data_0dB[list_norm_train,:],normal_data_C6dB[list_norm_train,:])) # 300 normal 
    x_train_06=np.vstack((x_train_abnormal,x_train_normal))
    label_train_06= np.hstack((np.tile(1,len(x_train_abnormal)),np.tile(0,len(x_train_normal))))
    x_test_abnormal=np.vstack((abnormal_data_6dB[list_abn_test,:],abnormal_data_0dB[list_abn_test,:],abnormal_data_C6dB[list_abn_test,:]))
    x_test_normal=np.vstack((normal_data_6dB[list_norm_test,:],normal_data_0dB[list_norm_test,:],normal_data_C6dB[list_norm_test,:]))
    x_test_06=np.vstack((x_test_abnormal,x_test_normal))
    label_test_06=np.hstack((np.tile(1,len(x_test_abnormal)),np.tile(0,len(x_test_normal))))
    X_train = x_train_06.astype('float32')
    X_test=x_test_06.astype('float32')
    label_test=label_test_06
    svm_train= X_train
    svm_test=X_test
    clf = SVC(kernel='linear', probability=True)
    clf.fit(svm_train, label_train_06)
    pred_label=clf.predict_proba(svm_test)

    score=pred_label
    t=0;
    split_fac=40
    num_segment=split_fac;
    pred_prob=[]
    for i in range(int(np.shape(X_test)[0]/split_fac)):
        pred=np.sum(score[t:t+num_segment,:],0)
        pred_prob.append(pred)
        t=t+num_segment
    pred_prob1=(np.asarray(pred_prob))/40
    y_class= pred_prob1.argmax(axis=-1)
    actual_labels=(label_test[0:np.shape(X_test)[0]:split_fac])
    auc=metrics.roc_auc_score(actual_labels,y_class)
    fpr, tpr, thresholds_p = metrics.roc_curve(actual_labels, pred_prob1[:,1], pos_label=1) #1 means abnormal
    fnr, tnr, thresholds_n = metrics.roc_curve(actual_labels, pred_prob1[:,0], pos_label=0)
    lst_p=np.arange(0,len(thresholds_p),len(thresholds_p)/100,dtype=int)
    lst_n=np.arange(0,len(thresholds_n),len(thresholds_n)/100,dtype=int)    
    lst_new_p=list_gen(thresholds_p,lst_p)
#    lst_new1_p=np.append(list_new_p,np.arange(len(thresholds_p)-10,len(thresholds_p),1,dtype=int))  
    lst_new_n=list_gen(thresholds_n,lst_n)
#    lst_new1_n=np.append(list_new_n,np.arange(len(thresholds_n)-10,len(thresholds_n),1,dtype=int))      
    AUC.append(auc)
    auc_iter=auc_iter+1
    print(auc,'  At trial ', auc_iter)    
    TPR.append(tpr[lst_new_p])
    FPR.append(fpr[lst_new_p])
    FNR.append(fnr[lst_new_n])
    TNR.append(tnr[lst_new_n])
    th_p.append(thresholds_p[lst_new_p])
    th_n.append(thresholds_n[lst_new_n])


TPR_all=np.average(TPR,0)
FPR_all= np.average(FPR,0)
TNR_all=np.average(TNR,0)
FNR_all=np.average(FNR,0)
Th_n=np.average(th_n,0)
Th_p=np.average(th_p,0)

np.save('20/TPR',TPR_all)
np.save('20/FPR',FPR_all)
np.save('20/TNR',TNR_all)
np.save('20/FNR',FNR_all)
np.save('20/AUC',AUC)
np.save('20/Th_p',Th_p)

print('T=20...completed and saved')
#%%


Num_examples=55
TPR=[]
FPR=[]
TNR=[]
FNR=[]
AUC=[]
th_p=[]
th_n=[]
auc_iter=0

for i in range(loop_iter):
    T=Num_examples
    list_abn=[]
    random.seed(i)
    for i in range(T):
        v=randint(0,int(len(abnormal_data_6dB)/40)-1)
        a=list(range(40*v,40*v+40))
        list_abn.append(a)
    list_norm=[]
    random.seed(i)
    for i in range(T):
        v=randint(0,int(len(normal_data_6dB)/40)-1)
        a=list(range(40*v,40*v+40))
        list_norm.append(a)
    list_abn_train=np.hstack(list_abn)
    list_norm_train=np.hstack(list_norm)
    list_abn_test=list(set(list(range(0,int(len(abnormal_data_6dB)))))-set(list_abn_train))
    list_norm_test=list(set(list(range(0,int(len(normal_data_6dB)))))-set(list_norm_train))
    x_train_abnormal=np.vstack((abnormal_data_6dB[list_abn_train,:],abnormal_data_6dB[list_abn_train,:],abnormal_data_C6dB[list_abn_train,:]))  #100 abnormal
    x_train_normal=np.vstack((normal_data_6dB[list_norm_train,:],normal_data_0dB[list_norm_train,:],normal_data_C6dB[list_norm_train,:])) # 300 normal 
    x_train_06=np.vstack((x_train_abnormal,x_train_normal))
    label_train_06= np.hstack((np.tile(1,len(x_train_abnormal)),np.tile(0,len(x_train_normal))))
    x_test_abnormal=np.vstack((abnormal_data_6dB[list_abn_test,:],abnormal_data_0dB[list_abn_test,:],abnormal_data_C6dB[list_abn_test,:]))
    x_test_normal=np.vstack((normal_data_6dB[list_norm_test,:],normal_data_0dB[list_norm_test,:],normal_data_C6dB[list_norm_test,:]))
    x_test_06=np.vstack((x_test_abnormal,x_test_normal))
    label_test_06=np.hstack((np.tile(1,len(x_test_abnormal)),np.tile(0,len(x_test_normal))))
    X_train = x_train_06.astype('float32')
    X_test=x_test_06.astype('float32')
    label_test=label_test_06
    svm_train= X_train
    svm_test=X_test
    clf = SVC(kernel='linear', probability=True)
    clf.fit(svm_train, label_train_06)
    pred_label=clf.predict_proba(svm_test)

    score=pred_label
    t=0;
    split_fac=40
    num_segment=split_fac;
    pred_prob=[]
    for i in range(int(np.shape(X_test)[0]/split_fac)):
        pred=np.sum(score[t:t+num_segment,:],0)
        pred_prob.append(pred)
        t=t+num_segment
    pred_prob1=(np.asarray(pred_prob))/40
    y_class= pred_prob1.argmax(axis=-1)
    actual_labels=(label_test[0:np.shape(X_test)[0]:split_fac])
    auc=metrics.roc_auc_score(actual_labels,y_class)
    fpr, tpr, thresholds_p = metrics.roc_curve(actual_labels, pred_prob1[:,1], pos_label=1) #1 means abnormal
    fnr, tnr, thresholds_n = metrics.roc_curve(actual_labels, pred_prob1[:,0], pos_label=0)
    lst_p=np.arange(0,len(thresholds_p),len(thresholds_p)/100,dtype=int)
    lst_n=np.arange(0,len(thresholds_n),len(thresholds_n)/100,dtype=int)    
    lst_new_p=list_gen(thresholds_p,lst_p)
#    lst_new1_p=np.append(list_new_p,np.arange(len(thresholds_p)-10,len(thresholds_p),1,dtype=int))  
    lst_new_n=list_gen(thresholds_n,lst_n)
#    lst_new1_n=np.append(list_new_n,np.arange(len(thresholds_n)-10,len(thresholds_n),1,dtype=int))      
    AUC.append(auc)
    auc_iter=auc_iter+1
    print(auc,'  At trial ', auc_iter)    
    TPR.append(tpr[lst_new_p])
    FPR.append(fpr[lst_new_p])
    FNR.append(fnr[lst_new_n])
    TNR.append(tnr[lst_new_n])
    th_p.append(thresholds_p[lst_new_p])
    th_n.append(thresholds_n[lst_new_n])


TPR_all=np.average(TPR,0)
FPR_all= np.average(FPR,0)
TNR_all=np.average(TNR,0)
FNR_all=np.average(FNR,0)
Th_n=np.average(th_n,0)
Th_p=np.average(th_p,0)

np.save('55/TPR',TPR_all)
np.save('55/FPR',FPR_all)
np.save('55/TNR',TNR_all)
np.save('55/FNR',FNR_all)
np.save('55/AUC',AUC)
np.save('55/Th_p',Th_p)

print('T=55...completed and saved')
