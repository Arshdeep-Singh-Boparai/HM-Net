import pickle
import librosa
#from scipy.misc import imread
import pickle
import sys


#%%

def load_audio(audio_path, sr=None):
    # By default, librosa will resample the signal to 22050Hz(sr=None). And range in (-1., 1.)
    sound_sample, sr = librosa.load(audio_path, sr=8000, mono=True)#,duration=10.0)

    return sound_sample, sr


#%% data read

audio_folder_path="path of audio files"

feature_save_folder='path to save features'
filename='data_machineID_Logmel'
os.chdir(feature_save_folder) 
data=[]
cl=0
labels=[]
for root, dirs, files in os.walk(audio_folder_path, topdown=False):
	for name in dirs:
		parts = []
		parts += [each for each in os.listdir(os.path.join(root,name)) if each.endswith('.wav')]
		print(name, "...")
		kj=0
		
		
		for part in parts:
			kj=kj+1
			sound_sample,sr =load_audio(os.path.join(root,name,part))
			mel_spectrogram = librosa.feature.melspectrogram(y = sound_sample,sr = sr, n_fft =400, hop_length =400,n_mels = 64, power = 2) #50ms window length
			log_mel_spectrogram = 20.0/2 * np.log10(mel_spectrogram + sys.float_info.epsilon)
			j=0
			for i in range(int((np.shape(log_mel_spectrogram)[1]/5))):
				data.append(log_mel_spectrogram[:,j:j+5].flatten())
				j=j+5
			print(part,'  feature saved  ',kj,'   ', np.shape(log_mel_spectrogram))
			labels=np.hstack((labels,np.tile(cl,int((np.shape(log_mel_spectrogram)[1]/5)))))
		cl=cl+1



np.save(filename,data)
np.save('labels_train_logmel',labels)
Shape_data=np.shape(data)        
print(Shape_data,' shape of data', np.shape(labels), 'labels shape',filename,'   saved')        
