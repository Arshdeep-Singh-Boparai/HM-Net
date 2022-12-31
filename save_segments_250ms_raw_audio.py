# -*- coding: utf-8 -*-
"""
Created on Sat Dec 31 02:31:51 2022

@author: Arshdeep Singh
"""


import pickle
import librosa
import pickle
import sys
#%%

def load_audio(audio_path, sr=None):
    # By default, librosa will resample the signal to 22050Hz(sr=None). And range in (-1., 1.)
    sound_sample, sr = librosa.load(audio_path, sr=8000, mono=True,duration=10.0)

    return sound_sample, sr

#%%..................................................................................

split_fac=40

feature_save_folder='path to save the segment wise raw audio in numpy format'
#%% data read



print(' train data 6db now processing....................................................................')


audio_folder_path="path of audio files"





filename='normal_data_6dB'
file_list=[]

os.chdir(feature_save_folder) 
data=[]
cl=0
labels=[]
for root, dirs, files in os.walk(audio_folder_path, topdown=False):
	for name in dirs:
		parts = []
		parts += [each for each in os.listdir(os.path.join(root,name)) if each.endswith('.wav')]
		print(name, "...")
		i=0
		
		for part in parts:
			sound_sample,sr =load_audio(os.path.join(root,name,part))
			sound_sample *= 256			
			example=np.array(sound_sample)
			sd=np.split(example,split_fac)
			data.append(sd)
			file_list.append(part)
			i=i+1
			print(part, str(i), '  feature saved  ',np.shape(sd))
			labels=np.hstack((labels,np.tile(cl,40)))
		cl=cl+1


data=np.vstack(data)
np.save('list_train_files',file_list)
np.save(filename,data)
np.save('labels',labels)
Shape_data=np.shape(data)        
print(Shape_data,' shape of data', np.shape(labels), 'labels shape',filename,'   saved')     
