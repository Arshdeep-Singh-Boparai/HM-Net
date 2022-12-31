# HM-Net

The repository contains sound-based health monitoring of industrial machines framework. Three feature reprsentation are used to represent sound signal recorded from industrial machines, (a) HM-Net: Health monitoring convolutional neural network (CNN) trained on industrial machines (b) SoundNet: A pre-trained neural network trained on large-scale dataset and (c) log-melspectrogram based time-frequency representations. After extracting features from the feature rerpesentation framework, a linear support vector machine (SVM) classifier is trained for classification.

A brief description of various python scripts is give below,

HMNet_SVM.py:  Feature extraction using HM-Net and SVM classification.

