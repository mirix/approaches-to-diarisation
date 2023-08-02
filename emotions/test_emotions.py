# De Bruyne L, De Clercq O, Hoste V. 
# Mixing and Matching Emotion Frameworks: 
# Investigating Cross-Framework Transfer Learning for Dutch Emotion Detection. 
# Electronics. 2021; 10(21):2643. 
# https://doi.org/10.3390/electronics10212643

# Mehrabian A, Russell JA. An approach to environmental psychology. the MIT Press; 1974.
# https://en.wikipedia.org/wiki/PAD_emotional_state_model

# Paul Ekman (1992) An argument for basic emotions, Cognition and Emotion, 6:3-4, 169-200, 
# DOI: 10.1080/02699939208411068 

# v / p = valence / pleasure -> The Pleasure-Displeasure Scale measures how pleasant or unpleasant one feels about something.
# a = arousal -> The Arousal-Nonarousal Scale measures how energized or soporific one feels. It is not the intensity of the emotion.
# d = dominance -> The Dominance-Submissiveness Scale represents the controlling and dominant versus controlled or submissive one feels. 


import os
# This version is CPU-only, you will need to adapt it for GPU
# it will used all the cores detected
n_cores = str(os.cpu_count())
os.environ['OMP_NUM_THREADS'] = n_cores
os.environ['MKL_NUM_THREADS'] = n_cores

import shutil
shutil.rmtree('tmp', ignore_errors=True)
if not os.path.isdir('tmp'):
	os.makedirs('tmp')

import srt
import pandas as pd
from functools import reduce
from sklearn.metrics.pairwise import pairwise_distances
import numpy as np

import sox
import librosa
import torch
import torch.nn as nn
from transformers import Wav2Vec2Processor
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2Model,
    Wav2Vec2PreTrainedModel,
)

def secondsToStr(t):
    return "%02d:%02d:%02d,%03d" % \
        reduce(lambda ll,b : divmod(ll[0],b) + ll[1:],
            [(round(t*1000),),1000,60,60])

class RegressionHead(nn.Module):
    r"""Classification head."""

    def __init__(self, config):

        super().__init__()

        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):

        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)

        return x

class EmotionModel(Wav2Vec2PreTrainedModel):
    r"""Speech emotion classifier."""

    def __init__(self, config):

        super().__init__(config)

        self.config = config
        self.wav2vec2 = Wav2Vec2Model(config)
        self.classifier = RegressionHead(config)
        self.init_weights()

    def forward(
            self,
            input_values,
    ):

        outputs = self.wav2vec2(input_values.reshape(1, input_values.size(dim=0)))
        #outputs = self.wav2vec2(input_values)
        hidden_states = outputs[0]
        hidden_states = torch.mean(hidden_states, dim=1)
        logits = self.classifier(hidden_states)

        return hidden_states, logits
        
def process_func(
    x: np.ndarray,
    sampling_rate: int,
    embeddings: bool = False,
) -> np.ndarray:
    r"""Predict emotions or extract embeddings from raw audio signal."""

    # run through processor to normalize signal
    # always returns a batch, so we just get the first entry
    # then we put it on the device
    y = processor(x, sampling_rate=sampling_rate)
    y = y['input_values'][0]
    y = torch.from_numpy(y).to(device)

    # run through model
    with torch.no_grad():
        y = model(y)[0 if embeddings else 1]

    # convert to numpy
    y = y.detach().cpu().numpy()

    return y

# load model from hub
device = 'cpu'
model_name = 'audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim'
processor = Wav2Vec2Processor.from_pretrained(model_name)
model = EmotionModel.from_pretrained(model_name)

# Ekman basic emotions
emotions = ['anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise']
# Mehrabian/Russell emotion dimensions
vad = ['a', 'd', 'v']

# Centroids of the Ekman emotions in a VAD diagram
anger = {'v': -0.51, 'a': 0.59, 'd': 0.25}
disgust = {'v': -0.60, 'a': 0.35, 'd': 0.11}
fear = {'v': -0.64, 'a': 0.60, 'd': -0.43}
joy = {'v': 0.76, 'a': 0.48, 'd': 0.35}
sadness = {'v': -0.63, 'a': -0.27, 'd': -0.33}
surprise = {'v':0.40, 'a': 0.67, 'd': -0.13}

df = pd.DataFrame(index=vad)

for emotion in emotions:
	df[emotion] = df.index.map(locals()[emotion])
	
matrix = pairwise_distances(df.T.values, Y=df.T.values, metric='euclidean', n_jobs=None, force_all_finite=True)

df_matrix = pd.DataFrame(matrix, columns=emotions, index=emotions)

#print(df_matrix)

# Use lower 25% quatile as threshold
quantile = df_matrix.melt().value.quantile(0.22)
#print(quantile)

# sample signal
srt_path = '../diarealsamples'

#input_file = '/home/emoman/Work/diarisation/Irate_Customer_Service_call_Fyb2AiF1feI_converted.wav'

for wavs in os.scandir(srt_path):
	if wavs.is_file() and wavs.path.endswith('.wav'):
		
		base_name = wavs.name[:-4]
		print(base_name)
		
		conv_audio = srt_path + '/' + base_name + '.wav'
		subtitle = srt_path + '/' + base_name + '.srt'
		
		srt_file = open(subtitle, 'r')
		chunk_list = []
		for sub in srt.parse(srt_file):
			start = float(str(sub.start.seconds) + '.' + str(sub.start.microseconds))
			end = float(str(sub.end.seconds) + '.' + str(sub.end.microseconds))
			text = sub.content
			row = (text, start, end)
			chunk_list.append(row)
		
		df_transcript = pd.DataFrame(chunk_list, columns = ['Text', 'Start', 'End'])
		df_transcript[['Speaker', 'Text']] = df_transcript['Text'].apply(lambda x: pd.Series(str(x).split(']: ')))
		df_transcript['Speaker'] = df_transcript['Speaker'].str.replace('[', '', regex=False)
		df_transcript['Length'] = df_transcript['Text'].str.split().str.len()
		df_transcript['Duration'] = df_transcript['End'] - df_transcript['Start']
		#df_long = df_transcript[(df_transcript['Length'] > 5)].copy()
		
		chunk_list = []
		for index, row in df_transcript.iterrows():
			try:
				tfm = sox.Transformer()
				#cbn = sox.Combiner()
				tfm.set_globals(verbosity=1)
				tfm.set_output_format(rate=16000, channels=1)	
				tfm.trim(row['Start'], row['End'])
				tfm.silence(location=0)
				tmp_audio = 'tmp/tmp.wav'
				tfm.build_file(conv_audio, tmp_audio)
			
				signal, sampling_rate = librosa.load(tmp_audio, sr=16000, mono=True)
				
				os.remove(tmp_audio)
				
				#sampling_rate = 16000
				#signal = np.zeros((1, sampling_rate), dtype=np.float32)
				
				vad_sample = pd.DataFrame((process_func(signal, sampling_rate)[0]), columns=['vad_sample'], index=vad)
				
				# 0 1 scale to -1 1 scale
				vad_sample = (vad_sample - .5) * 2
				
				#print(vad_sample)
				
				#print(process_func(signal, sampling_rate, embeddings=True))
				# Pooled hidden states of last transformer layer
				
				###
					
				smatrix = pairwise_distances(df.T.values, Y=vad_sample.T.values, metric='euclidean', n_jobs=None, force_all_finite=True)
				
				df_smatrix = pd.DataFrame(smatrix, columns=['vad_sample'], index=emotions)
				
				#print(df_smatrix)
				
				# Penalty to distances above threshold
				
				if df_smatrix[df_smatrix['vad_sample'] < quantile].empty:
					df_smatrix.loc['neutral'] = 0
					df_smatrix = df_smatrix * quantile * 10
				else:
					df_smatrix.loc['neutral'] = quantile
					df_smatrix[df_smatrix >= quantile] = df_smatrix * quantile * 10
				
				# Convert distances to percentual scores
				df_scores = round(100 * np.exp(-df_smatrix).apply(lambda x: x/x.sum(), axis=0)).astype('int')
				#df_scores = df_scores[df_scores['vad_sample'] >= 25].copy()
				df_scores = df_scores[df_scores['vad_sample'] == df_scores['vad_sample'].max()].copy()
				
				#print(df_scores)
				#print(df_scores.to_dict()['vad_sample'])
				
				chunk_list.append(df_scores.to_dict()['vad_sample'])
			
			except:
				chunk_list.append({'neutral': 0})
			
		df_transcript['emotion'] = chunk_list
		
		df_transcript['Text_Emotion'] = '[' + df_transcript['Speaker'] + ' ' + df_transcript['emotion'].astype('string') + ']: ' + df_transcript['Text']

		with open(srt_path + '/' + base_name + '_EMO.srt', 'w', encoding = 'utf-8') as f:
			for ind, col in df_transcript.iterrows():
				ind = ind + 1
				f.write(str(ind) + '\n')
				f.write(secondsToStr(col['Start']) + ' --> ' + secondsToStr(col['End']) + '\n')
				f.write(str(col['Text_Emotion']) + '\n\n')

shutil.rmtree('tmp', ignore_errors=True)
