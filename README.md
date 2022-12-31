# HM-Net

The repository contains sound-based health monitoring of industrial machines framework. The experiments are performed on MIMII [1] and ToyADMOS [2] datasets. 

Three feature reprsentation are used to represent sound signal recorded from industrial machines, (a) HM-Net: Health monitoring convolutional neural network (CNN) trained on industrial machines (b) SoundNet [3]: A pre-trained neural network trained on large-scale dataset and (c) log-melspectrogram based time-frequency representations. After extracting features from the feature rerpesentation framework, a linear support vector machine (SVM) classifier is trained for classification.

A brief description of various python scripts is give below,

1. HMNet_SVM.py:  Feature extraction using HM-Net and SVM classification.

2. save_segments_250ms_raw_audio.py: Save 250ms sound segments in numpy format.

3. Log-mel.py: Extract log-mel spectrogram time-frquency representations for machine sounds. Please use the script HMNet_SVM.py by replacing HM-Net features with Log-mel features for SVM classification.

4. SoundNet_feature_extraction.py: Feature extraction using pre-trained SoundNet from raw-audio segments. Please use the script HMNet_SVM.py by replacing HM-Net features with SoundNet features for SVM classification.


References:


[1] Purohit, Harsh, et al. "MIMII DATASET: SOUND DATASET FOR MALFUNCTIONING INDUSTRIAL MACHINE INVESTIGATION AND INSPECTION." Acoustic Scenes and Events 2019 Workshop (DCASE2019). 2019.
[2] Koizumi, Yuma, et al. "ToyADMOS: A dataset of miniature-machine operating sounds for anomalous sound detection." 2019 IEEE Workshop on Applications of Signal Processing to Audio and Acoustics (WASPAA). IEEE, 2019.
[3] Aytar, Yusuf, Carl Vondrick, and Antonio Torralba. "Soundnet: Learning sound representations from unlabeled video." Advances in neural information processing systems 29 (2016).

Acknowldgement: 

The work is a part of internship work done at Inter Bangalore, India.
