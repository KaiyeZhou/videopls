
# coding: utf-8

# ## Build Audio Vectors
# Now that the labels have been extracted, we'll use the compiled csv (df_iemocap.csv) to split the original wav files into multiple frames

# In[1]:


# Try for one file first
import librosa
import os
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.style as ms
from tqdm import tqdm
import pickle

import IPython.display
import librosa.display
ms.use('seaborn-muted')
get_ipython().magic(u'matplotlib inline')


# In[2]:


file_path = '/home/zhouky00/multimodal-speech-emotion-recognition/data/Session1/dialog/wav/Ses01F_impro01.wav'

y, sr = librosa.load(file_path, sr=44100)
y, sr


# ## Loop through all the files

# In[3]:


import pandas as pd
import math

labels_df = pd.read_csv('/home/zhouky00/multimodal-speech-emotion-recognition/data/pre-processed/df_iemocap.csv')
iemocap_dir = '/home/zhouky00/multimodal-speech-emotion-recognition/data/'


# The following cells take some time until completely executed

# In[5]:


sr = 44100
audio_vectors = {}
# for sess in [5]:  # using one session due to memory constraint, can replace [5] with range(1, 6)
#     wav_file_path = '{}Session{}/dialog/wav/'.format(iemocap_dir, sess)
#     orig_wav_files = os.listdir(wav_file_path)
#     for orig_wav_file in tqdm(orig_wav_files):
#         try:
#             orig_wav_vector, _sr = librosa.load(wav_file_path + orig_wav_file, sr=sr)
#             orig_wav_file, file_format = orig_wav_file.split('.')
#             for index, row in labels_df[labels_df['wav_file'].str.contains(orig_wav_file)].iterrows():
#                 start_time, end_time, truncated_wav_file_name, emotion, val, act, dom = row['start_time'], row['end_time'], row['wav_file'], row['emotion'], row['val'], row['act'], row['dom']
#                 start_frame = math.floor(start_time * sr)
#                 end_frame = math.floor(end_time * sr)
#                 truncated_wav_vector = orig_wav_vector[start_frame:end_frame + 1]
#                 audio_vectors[truncated_wav_file_name] = truncated_wav_vector
#         except:
#             print('An exception occured for {}'.format(orig_wav_file))
#     with open('data/pre-processed/audio_vectors_{}.pkl'.format(sess), 'wb') as f:
#         pickle.dump(audio_vectors, f)

# for sess in [5]:  # using one session due to memory constraint, can replace [5] with range(1, 6)
sess = 1
wav_file_path = '{}Session{}/dialog/wav/'.format(iemocap_dir, sess)
orig_wav_files = os.listdir(wav_file_path)
for orig_wav_file in tqdm(orig_wav_files):
    try:
        orig_wav_vector, _sr = librosa.load(wav_file_path + orig_wav_file, sr=sr)
        orig_wav_file, file_format = orig_wav_file.split('.')
        for index, row in labels_df[labels_df['wav_file'].str.contains(orig_wav_file)].iterrows():
            start_time, end_time, truncated_wav_file_name, emotion, val, act, dom = row['start_time'], row['end_time'], row['wav_file'], row['emotion'], row['val'], row['act'], row['dom']
            start_frame = math.floor(start_time * sr)
            end_frame = math.floor(end_time * sr)
            truncated_wav_vector = orig_wav_vector[start_frame:end_frame + 1]
            audio_vectors[truncated_wav_file_name] = truncated_wav_vector
    except:
        print('An exception occured for {}'.format(orig_wav_file))
with open('/home/zhouky00/multimodal-speech-emotion-recognition/data/pre-processed/audio_vectors_{}.pkl'.format(sess), 'wb') as f:
    pickle.dump(audio_vectors, f)

