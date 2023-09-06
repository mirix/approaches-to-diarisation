### Clustering based diarisation script ###
# It expects to find the audio files in mp3 format
# in a folder called samples

### IMPORTS ###

# Generic libraries: no install needed

import warnings
warnings.filterwarnings('ignore')

from pathlib import Path

import os
n_cores = str(os.cpu_count())
os.environ['OMP_NUM_THREADS'] = n_cores
os.environ['MKL_NUM_THREADS'] = n_cores
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1,0"

from functools import reduce
import shutil
import shlex
import re

# Miscelaneus libraries: install needed

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
from nemo.collections.nlp.models import PunctuationCapitalizationModel
from sklearn import preprocessing
le = preprocessing.LabelEncoder()

import logging
logging.getLogger('nemo_logger').setLevel(logging.ERROR)

### Create working directories ###

if not os.path.isdir('diarealsamples'):
	os.makedirs('diarealsamples')

if not os.path.isdir('tmp'):
	os.makedirs('tmp')

### Define Whisper model ###
# for transcription #

device = 'cuda'
model_size = 'large-v2'

modelw = stable_whisper.load_model(model_size).to(device)
modelpc = PunctuationCapitalizationModel.from_pretrained('punctuation_en_bert').to(device)
modelsp = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained('nvidia/speakerverification_en_titanet_large').to(device)

### Functions ###

# Seconts to SRT		
def secondsToStr(t):
    return "%02d:%02d:%02d,%03d" % \
        reduce(lambda ll,b : divmod(ll[0],b) + ll[1:],
            [(round(t*1000),),1000,60,60])

# Compute Titanet embeddings            
def compute_embedding(row):
	tfm = sox.Transformer()
	tfm.set_globals(verbosity=1)
	tfm.set_output_format(rate=16000, channels=1)
	tfm.trim(row['Start'], row['End'])
	tfm.build_file(conv_audio, 'tmp/tmp.wav')
	# added .cpu() because now I am running this on GPUs
	# remove if problematic
	embedding = modelsp.get_embedding('tmp/tmp.wav').cpu()
	os.remove('tmp/tmp.wav')
	return embedding
	
# Repuctuation and recapitalisation
def repunct_recap(text):
	repunct = {'.,': '.', ',,': ',', ',.': ',', '.,.': '.', ',?': '?', ',,,':',', '.,,': '.', ',!': '!', '..': '.', '..': '...', ' –,': '.', ',–': '.'}
	punct_text = modelpc.add_punctuation_capitalization([text])[0]
	for key, val in repunct.items():
		punct_text = punct_text.replace(key, val)
	return punct_text

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
		print(' ')
		print(base_name, ':', sox.file_info.duration(audios), 'seconds')
		length.append(sox.file_info.duration(audios))
		demucs.separate.main(shlex.split('--two-stems vocals -n mdx_extra ' + 'samples/' + name + ' -o tmp'))
		rawsound = AudioSegment.from_file('tmp/mdx_extra/' + base_name + '/vocals.wav', 'wav') 
		rawsound = rawsound.set_channels(1) 
		rawsound = rawsound.set_frame_rate(16000)
		normalizedsound = effects.normalize(rawsound)  
		normalizedsound.export(conv_audio, format='wav')

### TRANSCRIBE ###

# Max sentence length (approx)
max_length = 44
# Sentences shorter than 50 words are split on the following characters
stops = ('。', '．', '.', '！', '!', '?', '？')
# For longer sentences, the comma is also a splitting mark
extra_stops = (',', '，')
# The following abbreviations are excluded
abbre = ('Dr.', 'Mr.', 'Mrs.', 'Ms.', 'vs.', 'Prof.', 'i.e.')

for wavs in os.scandir('diarealsamples'):
	base_name = wavs.name[:-4]
	if wavs.is_file() and wavs.path.endswith('.wav'):
		
		base_name = wavs.name[:-4]
		conv_audio = 'diarealsamples/' + base_name + '.wav'
		
		### Transcription ###
		
		result = modelw.transcribe(conv_audio, regroup='sp=.* /。/?/？/．/!/！')
		results = result.to_dict()['segments']

		### Sentence splitting ### 
		
		word_list = []
		start_list = []
		end_list =[]
		for segment in results:
			for word in segment['words']:
				word_list.append(word['word'])
				start_list.append(word['start'])
				end_list.append(word['end'])
				
		full_text = ''.join([str(i) for i in word_list])
		full_text = repunct_recap(full_text)

		chunk_list = []
		for i, word in enumerate(full_text.split()):
			if i == 0:
				start0 = start_list[i]
				end0 = end_list[i]
				word0 = word
				if word0.endswith(stops) and not word0.endswith(abbre):
					chunk_list.append((word0, start0, end0))
			else:
				if len(word0.split()) <= max_length:
					if not word0.endswith(stops) or word0.endswith(abbre):
						word1 = word
						word0 = word0 + ' ' + word1
						start0 = start0
						end0 = end_list[i]
						if word0.endswith(stops) and not word0.endswith(abbre):
							chunk_list.append((word0, start0, end0))
					else:
						word0 = word
						start0 = start_list[i]
						end0 = end_list[i]
						if word0.endswith(stops) and not word0.endswith(abbre):
							chunk_list.append((word0, start0, end0))						
				if len(word0.split()) > max_length:
					if not word0.endswith(stops + extra_stops) or word0.endswith(abbre):
						word1 = word
						word0 = word0 + ' ' + word1
						start0 = start0
						end0 = end_list[i]
						if word0.endswith(stops + extra_stops) and not word0.endswith(abbre):
							chunk_list.append((word0, start0, end0))
					else:
						word0 = word
						start0 = start_list[i]
						end0 = end_list[i]
						if word0.endswith(stops + extra_stops) and not word0.endswith(abbre):
							chunk_list.append((word0, start0, end0))
		
		df_transcript = pd.DataFrame(chunk_list, columns = ['Text', 'Start', 'End'])
		
		#df_transcript['Text'] = df_transcript['Text'].apply(repunct_recap)
		df_transcript['Text'] = df_transcript['Text'].map(lambda l: l[:1].upper() + l[1:])
		
		df_transcript['Duration'] = df_transcript['End'] - df_transcript['Start']
		df_transcript['Length'] = df_transcript['Text'].str.split().str.len()

		### DIARISE ###
		
		print(' ')
		print(base_name)
		
		### Remove zero-lengh segments and negative segments ###
		
		df_transcript = df_transcript[(df_transcript['Duration'] > 0)].copy()
		df_transcript = df_transcript.reset_index(drop=True)
		
		### Identify the indexes of long and short sentences ###
		seq_len = 5
		df_long = df_transcript[(df_transcript['Length'] > seq_len)].copy()
		df_shor = df_transcript[(df_transcript['Length'] <= seq_len)].copy()
		
		long_seg = list(df_long.index.values)
		shor_seg = list(df_shor.index.values)	
		
		# HDBSCAN dimension limit is typically 50-100
		# we go with 50
		
		dimensions = 50
		
		#############################################################################
		### If there are few long sentences we cluster all the sentences together ###
		#############################################################################
		
		if len(df_transcript) < 2:
			df_transcript['cluster_group'] = 1
			
		if len(long_seg) < 10 and len(df_transcript) >= 2:
			
			### Compute the embeddings of each sentence ###
			
			embeddings = df_transcript.apply(compute_embedding, axis=1)
			
			### Create am all-vs-all matrix of cosine distances between embeddings ###
			
			dist_matrix = []
			for emb in embeddings:
				row = []
				for emb1 in embeddings:
					try:
						distance = cdist(emb, emb1, metric='cosine')[0][0]
					except Exception:
						distance = 2
					row.append(distance)
				dist_matrix.append(row)
			
			df_dist = pd.DataFrame(dist_matrix)
			df_dist = df_dist.fillna(2)
			
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
			if comp < 1:
				comp = 1
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
			
			try:
				clusterable_embedding = umap.UMAP(
				    n_neighbors=n_neighbors,
				    min_dist=.0,
				    n_components=3,
				    random_state=31416,
				    metric='cosine',
				    #init='random'
				).fit_transform(df_dist)
			except ValueError:
				clusterable_embedding = df_dist
	
			# Another one with 50 or so for clustering
			
			try:
				clusterable_embedding_large = umap.UMAP(
				    n_neighbors=n_neighbors,
				    min_dist=.0,
				    n_components=comp,
				    random_state=31416,
				    metric='cosine',
				).fit_transform(df_dist)
			except ValueError:
				clusterable_embedding_large = df_dist
			
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
			
			### Reclustering of short sentences ###
			
			# Short sentences are sometimes misattributed
			# We take as reference the long sentences on each cluster
			# and reassign the short sentences to each of those clusters
			# by measuring the average distance
			
			df_reassign = df_dist.copy()
			df_reassign['Labels'] = clustered
			
			df_transcript['cluster_group'] = clustered
			
			if len(shor_seg) >= 1:
				for i in shor_seg:
					try:
						label_matrix = df_reassign.iloc[long_seg].groupby(['Labels'], as_index=False)[i].mean()
						new_label = int(label_matrix.loc[label_matrix[i].idxmin()]['Labels'])
						df_transcript.at[i, 'cluster_group'] = new_label
					except Exception:
						pass
		
		##########################################################################################			
		### If there are many long sentences we cluster only those and then recluster the rest ###
		##########################################################################################
		
		else:
			
			### Compute the embeddings of each sentence ###
			
			embeddings_long = df_long.apply(compute_embedding, axis=1)
			
			### Create am all-vs-all matrix of cosine distances between embeddings ###
			
			dist_matrix = []
			for emb in embeddings_long:
				row = []
				for emb1 in embeddings_long:
					try:
						distance = cdist(emb, emb1, metric='cosine')[0][0]
					except Exception:
						distance = 2
					row.append(distance)
				dist_matrix.append(row)
			
			df_dist = pd.DataFrame(dist_matrix)
			df_dist = df_dist.fillna(2)
			
			# If the matrix has more than 50 dimensions
			# we reduce to 50, otherwise to the 
			# number of dimensions minus 2
			# HDBSCAN seems to prefer UMAP embeddings to raw data
			
			rows = len(df_long.index)
			
			utter = rows
			print(utter)
			if utter >= dimensions + 2:
				comp = dimensions
			else:
				comp = utter - 2	
			if comp < 1:
				comp = 1
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
			
			try:
				clusterable_embedding = umap.UMAP(
				    n_neighbors=n_neighbors,
				    min_dist=.0,
				    n_components=3,
				    random_state=31416,
				    metric='cosine',
				    #init='random'
				).fit_transform(df_dist)
			except ValueError:
				clusterable_embedding = df_dist
	
			# Another one with 50 or so for clustering
			
			try:
				clusterable_embedding_large = umap.UMAP(
				    n_neighbors=n_neighbors,
				    min_dist=.0,
				    n_components=comp,
				    random_state=31416,
				    metric='cosine',
				).fit_transform(df_dist)
			except ValueError:
				clusterable_embedding_large = df_dist
			
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
			
			### Reclustering of short sentences ###
			# We reassign the short sentences to each of the long
			# sentence clusters by computing the average distance
			
			df_reassign = df_dist.copy()
			df_reassign['Labels'] = clustered
			
			df_long['cluster_group'] = clustered
			
			if len(shor_seg) >= 1:
				embeddings_short = df_shor.apply(compute_embedding, axis=1)
				
				embeddings_df = embeddings_long.to_frame()
				embeddings_df.columns = ['embeddings']
				embeddings_df['cluster_group'] = clustered
				
				emb_short_list = []
				for embedding in embeddings_short:
					average0 = 2
					for name, group in embeddings_df.groupby(['cluster_group']):
						emb_long_list = group['embeddings'].values.tolist()
						distance_list = []
						for emb in emb_long_list:
							distance = cdist(embedding, emb, metric='cosine')[0][0]
							distance_list.append(distance)
						average = sum(distance_list) / len(distance_list)
						if average <= average0:
							average0 = average
							cluster_name = name[0]
					emb_short_list.append(cluster_name)
				
				df_shor['cluster_group'] = emb_short_list
				
				df_transcript = pd.concat([df_long, df_shor], sort=False).sort_index()										
			else:
				df_transcript = df_long.copy()
						
		########################################################################################################
			
		df_transcript['cluster_group'] = le.fit_transform(df_transcript['cluster_group'])
		df_transcript['cluster_group'] = df_transcript['cluster_group'] + 1
		
		clustered = df_transcript['cluster_group'].tolist()
		
		# Some cluster quality indicators
		
		print('Clusters: ' + ' '.join(str(i) for i in list(set(clustered))))
		print('Ratio of data points labelled: ' + str((np.sum(clustered) / rows).round(2)))
		print('Cluster quality (1 is perfect): ' + str((clusterer.relative_validity_).round(2)))
		print(' ')
	
		### PLOT ###
		# Interactive 3D plots HTML
		
		# We prepare the data for plotting
		
		if clusterable_embedding is not None:
			umap_df = pd.DataFrame(data = clusterable_embedding, columns = ['x', 'y', 'z'])
			
			df_umap = pd.concat([df_transcript, umap_df], axis=1)
			df_umap['cluster_group'] = clustered
			
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
		
		with open('diarealsamples/' + base_name + '.srt', 'w', encoding = 'utf-8') as f:
			for ind, col in df_transcript.iterrows():
				ind = ind + 1
				f.write(str(ind) + '\n')
				f.write(secondsToStr(col['Start']) + ' --> ' + secondsToStr(col['End']) + '\n')
				f.write('[SPEAKER ' + str(col['cluster_group']) + ']:' + ' ' + str(col['Text']) + '\n\n')

### DELETE TEMP DIR ###

shutil.rmtree('tmp', ignore_errors=True)
