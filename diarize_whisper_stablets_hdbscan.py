### Clustering based diarisation script ###
# It expects to find the audio files in mp3 format
# in a folder called samples

### IMPORTS ###

# Generic libraries no install needed

from pathlib import Path

import timeit
start_time = timeit.default_timer()

import os
# This version is CPU-only, you will need to adapt it for GPU
# it will used all the cores detected
n_cores = str(os.cpu_count())
os.environ['OMP_NUM_THREADS'] = n_cores
os.environ['MKL_NUM_THREADS'] = n_cores

from functools import reduce
import shutil
import shlex
import re

# Miscelaneus libraries install needed

from scipy.spatial.distance import cdist
import plotly.express as px
import pandas as pd
import numpy as np

# Audio-related libraries

import demucs.separate
from pydub import AudioSegment, effects  
import sox

# Clustering-related libraries

import umap
import hdbscan

# Machine learning libraries

import torch
import stable_whisper
import nemo.collections.asr as nemo_asr
from sklearn import preprocessing
le = preprocessing.LabelEncoder()

### Create working directories ###

if not os.path.isdir('diarealsamples'):
	os.makedirs('diarealsamples')

if not os.path.isdir('tmp'):
	os.makedirs('tmp')


### Define Whisper model ###
# for transcription #

model_size = 'large-v2'
modelw = stable_whisper.load_model(model_size)
modelsp = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained('nvidia/speakerverification_en_titanet_large')

### PREPROCESS AUDIO ###
# Vocal isolation is commented out as is it time-consuming
# and not required for the current project

ext = ('.mp3')
length = []
for audios in os.scandir('samples'):
	name = audios.name
	base_name = name[:-4]
	conv_audio = 'diarealsamples/' + base_name + '.wav'
	if (audios.is_file() and audios.path.endswith(ext) and not Path(conv_audio).is_file()):
		print(base_name, ':', sox.file_info.duration(audios), 'seconds')
		length.append(sox.file_info.duration(audios))
		demucs.separate.main(shlex.split('--two-stems vocals -n mdx_extra ' + 'samples/' + name + ' -o tmp'))
		rawsound = AudioSegment.from_file('tmp/mdx_extra/' + base_name + '/vocals.wav', 'wav') 
		rawsound = rawsound.set_channels(1) 
		rawsound = rawsound.set_frame_rate(16000)
		normalizedsound = effects.normalize(rawsound)  
		normalizedsound.export(conv_audio, format='wav')

### TRANSCRIBE ###

stops = ('。', '．', '.', '！', '!', '?', '？')
abbre = ('Dr.', 'Mr.', 'Mrs.', 'Ms.', 'vs.', 'Prof.')

dummy_list = []
for wavs in os.scandir('diarealsamples'):
	base_name = wavs.name[:-4]
	if wavs.is_file() and wavs.path.endswith('.wav'):
		
		base_name = wavs.name[:-4]
		print(base_name)
		conv_audio = 'diarealsamples/' + base_name + '.wav'
		
		### Transcription ###
		
		result = modelw.transcribe(conv_audio, regroup='sp=.* /。/?/？/．/!/！')
		results = result.to_dict()['segments']
		
		### Post-processing for regrouping sentences ###
		
		text0 = results[0]['text']
		start0 = results[0]['start']
		end0 = results[0]['end']
		chunk_list = []
		if text0.endswith(stops) and not text0.endswith(abbre):
			chunk_list.append((text0, start0, end0))
		
		l = len(results)
		
		for i in range(1, l):
			segment = result.to_dict()['segments'][i]
			if text0.endswith(stops) and not text0.endswith(abbre):
				text0 = segment['text']
				start0 = segment['start']
				end0 = segment['end']
				if text0.endswith(stops) and not text0.endswith(abbre):
					chunk_list.append((text0, start0, end0))
			else:
				text = segment['text']
				start = segment['start']
				end = segment['end']
				text0 = text0 + text
				start0 = start0
				end0 = end
				if text0.endswith(stops) and not text0.endswith(abbre):
					chunk_list.append((text0, start0, end0))
		
		df_transcript = pd.DataFrame(chunk_list, columns = ['Text', 'Start', 'End'])
		
		df_transcript['Duration'] = df_transcript['End'] - df_transcript['Start']
		df_transcript['Length'] = df_transcript['Text'].str.split().str.len()


### DIARISE ###

		print(' ')
		print(base_name)
		#conv_audio = base_name + '.wav'
		
		### Remove zero-lengh segments and negative segments ###
		
		df_transcript = df_transcript[(df_transcript['Duration'] > 0)].copy()
		df_transcript = df_transcript.reset_index(drop=True)
		
		### Identify the indexes of long and short sentences ###
		
		df_long = df_transcript[(df_transcript['Length'] > 5)]
		df_shor = df_transcript[(df_transcript['Length'] <= 5)]
		
		long_seg = list(df_long.index.values)
		shor_seg = list(df_shor.index.values)	
		
		### Compute the embeddings of each sentence ###
		
		def compute_embedding(row):
			tfm = sox.Transformer()
			tfm.set_globals(verbosity=1)
			tfm.set_output_format(rate=16000, channels=1)
			tfm.trim(row['Start'], row['End'])
			tfm.build_file(conv_audio, 'tmp/tmp.wav')
			try:
				embedding = modelsp.get_embedding('tmp/tmp.wav').cpu()
				os.remove('tmp/tmp.wav')
				return embedding
			except:
				os.remove('tmp/tmp.wav')
				return np.nan
		
		embeddings = df_transcript.apply(compute_embedding, axis=1)
		
		### Create am all-vs-all matrix of cosine distances between embeddings ###
		
		dist_matrix = []
		for emb in embeddings:
			row = []
			for emb1 in embeddings:
				try:
					distance = cdist(emb, emb1, metric='cosine')[0][0]
				except:
					distance = 2
				row.append(distance)
			dist_matrix.append(row)
		
		df_dist = pd.DataFrame(dist_matrix)
		df_dist = df_dist.fillna(2)
		
		# HDBSCAN dimension limit is typically 50-100
		# we go with 50
		
		dimensions = 50
		
		# If the matrix has more than 50 dimensions
		# we reduce to 50, otherwise to the 
		# number of dimensions minus 2
		# HDBSCAN seems to prefer UMAP embeddings to raw data
		
		rows = len(df_transcript.index)
		
		utter = rows
		print(utter)
		if utter >= dimensions + 2:
			comp = dimensions
		else:
			comp = utter - 2
		print(comp)
		
		# We make the number of neighbors and the 
		# cluster size proportional to the matrix size
		# (division by 4)
		
		n_neighbors = rows // 4
		if n_neighbors < 2:
			n_neighbors = 2
		
		cluster_size = rows // 4
		if cluster_size < 3:
			cluster_size = 3
			
		# We create two different dimensionality reduction
		# embeddings
		
		# One with only 3 dimensions for plotting
		
		clusterable_embedding = umap.UMAP(
		    n_neighbors=n_neighbors,
		    min_dist=.0,
		    n_components=3,
		    random_state=31416,
		    metric='cosine',
		).fit_transform(df_dist)
		
		# Another one with 50 or so for clustering
		
		clusterable_embedding_large = umap.UMAP(
		    n_neighbors=n_neighbors,
		    min_dist=.0,
		    n_components=comp,
		    random_state=31416,
		    metric='cosine',
		).fit_transform(df_dist)
		
		# We define the clustering algorithm and cluster
		
		clusterer = hdbscan.HDBSCAN(
		    min_samples=1,
		    min_cluster_size=cluster_size,
		    cluster_selection_method='leaf',
		    cluster_selection_epsilon=2,
		    gen_min_span_tree=True,
		    prediction_data=True
		).fit(clusterable_embedding_large)
		
		labels = list(clusterer.labels_)
		clustered = labels
		
		# We prepare the data for plotting
		
		umap_df = pd.DataFrame(data = clusterable_embedding, columns = ['x', 'y', 'z'])
		umap_df['cluster_group'] = clustered
		
		df_umap = pd.concat([df_transcript, umap_df], axis=1)
		
		df_reassign = df_dist.copy()
		df_reassign['Labels'] = clustered
		
		### Reclustering of short sentences ###
		
		# Short sentences are sometimes misattributed
		# We take as reference the long sentenced on each cluster
		# and reassign the short sentences to each of those clusters
		# by measuring the average distance
		
		for i in shor_seg:
			label_matrix = df_reassign.iloc[long_seg].groupby(['Labels'], as_index=False)[i].mean()
			new_label = int(label_matrix.loc[label_matrix[i].idxmin()]['Labels'])
			df_umap.at[i, 'cluster_group'] = new_label
		
		df_umap['cluster_group'] = le.fit_transform(df_umap['cluster_group'])
		df_umap['cluster_group'] = df_umap['cluster_group'] + 1
		
		clustered = df_umap['cluster_group'].tolist()
		
		# Some cluster quality indicators
		
		print('Clusters: ' + ' '.join(str(i) for i in list(set(clustered))))
		print('Ratio of data points labelled: ' + str((np.sum(clustered) / rows).round(2)))
		print('Cluster quality (1 is perfect): ' + str((clusterer.relative_validity_).round(2)))
	
		### PLOT ###
		# Interactive 3D plots HTML
		
		fig = px.scatter_3d(
		    df_umap, x='x', y='y', z='z',
		    color=df_umap['cluster_group'],
		    color_continuous_scale=px.colors.sequential.Jet,
		    hover_data=['Text', 'Duration', 'Length', 'cluster_group']
		)
		fig.update_scenes(xaxis_visible=False, yaxis_visible=False, zaxis_visible=False )
		fig.update_traces(marker_size = 4)
		fig.write_html('diarealsamples/' + base_name + '_hdbscan.html')
		
		### SAVE DIARISED SUBTITLES ###
		
		def secondsToStr(t):
		    return "%02d:%02d:%02d,%03d" % \
		        reduce(lambda ll,b : divmod(ll[0],b) + ll[1:],
		            [(round(t*1000),),1000,60,60])
		
		with open('diarealsamples/' + base_name + '.srt', 'w', encoding = 'utf-8') as f:
			for ind, col in df_umap.iterrows():
				ind = ind + 1
				f.write(str(ind) + '\n')
				f.write(secondsToStr(col['Start']) + ' --> ' + secondsToStr(col['End']) + '\n')
				f.write('[SPEAKER ' + str(col['cluster_group']) + ']:' +  str(col['Text']) + '\n\n')

### TIME THE SCRIPT ###

time_processed = sum(length)
print('Total audo duration: ', time_processed, 'seconds')

stop_time = timeit.default_timer()
computing_time = stop_time - start_time
print('It took', computing_time, 'seconds.')

speed = time_processed / computing_time
print('Speed: ', speed, 'seconds per second')
print('Processing one minute of audio takes: ', (1 / speed) * 60 , 'seconds')


### DELETE TEMP DIR ###

shutil.rmtree('tmp', ignore_errors=True)
