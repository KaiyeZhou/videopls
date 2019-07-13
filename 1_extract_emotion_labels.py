
# coding: utf-8

# # Extract labels from the evaluation files
# 
# Test for one file first

# In[1]:


import re

# first test with one file
file_path = '/home/zhouky00/multimodal-speech-emotion-recognition/data/Session1/dialog/EmoEvaluation/Ses01F_impro01.txt'


# In[8]:


useful_regex = re.compile(r'\[.+\]\n', re.IGNORECASE)


# In[13]:


with open(file_path) as f:
    file_content = f.read()
    
info_lines = re.findall(useful_regex, file_content)


# In[20]:


for l in info_lines[1:10]:
    print(l.strip().split('\t'))


# ## Compile all the information in a single file

# In[64]:


#import re
import os


info_line = re.compile(r'\[.+\]\n', re.IGNORECASE)

start_times, end_times, wav_file_names, emotions, vals, acts, doms = [], [], [], [], [], [], []

# for sess in range(1, 6):
#     emo_evaluation_dir = '/home/zhouky00/multimodal-speech-emotion-recognition/data/Session{}/dialog/EmoEvaluation'.format(sess)
#     evaluation_files = [l for l in os.listdir(emo_evaluation_dir) if 'Ses' in l]
#     for file in evaluation_files:
#         with open(emo_evaluation_dir + file) as f:
#             content = f.read()
#         info_lines = re.findall(info_line, content)
#         for line in info_lines[1:]:  # the first line is a header
#             start_end_time, wav_file_name, emotion, val_act_dom = line.strip().split('\t')
#             start_time, end_time = start_end_time[1:-1].split('-')
#             val, act, dom = val_act_dom[1:-1].split(',')
#             val, act, dom = float(val), float(act), float(dom)
#             start_time, end_time = float(start_time), float(end_time)
#             start_times.append(start_time)
#             end_times.append(end_time)
#             wav_file_names.append(wav_file_name)
#             emotions.append(emotion)
#             vals.append(val)
#             acts.append(act)
#             doms.append(dom)

emo_evaluation_dir = '/home/zhouky00/multimodal-speech-emotion-recognition/data/Session1/dialog/EmoEvaluation'.format(sess)
evaluation_files = [l for l in os.listdir(emo_evaluation_dir) if 'Ses' in l]
for file in evaluation_files:
    with open(emo_evaluation_dir + file) as f:
        content = f.read()
    info_lines = re.findall(info_line, content)
    for line in info_lines[1:]:  # the first line is a header
        start_end_time, wav_file_name, emotion, val_act_dom = line.strip().split('\t')
        start_time, end_time = start_end_time[1:-1].split('-')
        val, act, dom = val_act_dom[1:-1].split(',')
        val, act, dom = float(val), float(act), float(dom)
        start_time, end_time = float(start_time), float(end_time)
        start_times.append(start_time)
        end_times.append(end_time)
        wav_file_names.append(wav_file_name)
        emotions.append(emotion)
        vals.append(val)
        acts.append(act)
        doms.append(dom)


# In[68]:


import pandas as pd

df_iemocap = pd.DataFrame(columns=['start_time', 'end_time', 'wav_file', 'emotion', 'val', 'act', 'dom'])

df_iemocap['start_time'] = start_times
df_iemocap['end_time'] = end_times
df_iemocap['wav_file'] = wav_file_names
df_iemocap['emotion'] = emotions
df_iemocap['val'] = vals
df_iemocap['act'] = acts
df_iemocap['dom'] = doms

df_iemocap.tail()


# In[72]:


df_iemocap.to_csv('/home/zhouky00/multimodal-speech-emotion-recognition/data/pre-processed/df_iemocap.csv', index=False)

