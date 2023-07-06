import os
n_cores = '6'
os.environ['OMP_NUM_THREADS'] = n_cores
os.environ['MKL_NUM_THREADS'] = n_cores

import numpy as np

import srt
import re
import pandas as pd
from functools import reduce

from scipy.spatial.distance import cdist

import torch
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
from pyannote.audio import Audio
from pyannote.core import Segment

import plotly.express as px

import umap
import hdbscan

from sklearn import preprocessing
le = preprocessing.LabelEncoder()

modelsp = PretrainedSpeakerEmbedding('nvidia/speakerverification_en_titanet_large', device=torch.device('cpu'))

for wavs in os.scandir('diarealsamples'):
	if wavs.is_file() and wavs.path.endswith('.wav'):
		
		base_name = wavs.name[:-4]
		print(' ')
		print(base_name)
		conv_audio = base_name + '.wav'
		sub_file = base_name + '_nodiarisation.srt'
		
		srt_file = open('diarealsamples/' + sub_file, 'r')
		
		chunk_list = []
		for sub in srt.parse(srt_file):
			start = float(str(sub.start.seconds) + '.' + str(sub.start.microseconds))
			end = float(str(sub.end.seconds) + '.' + str(sub.end.microseconds))
			text = sub.content
			row = (text, start, end)
			chunk_list.append(row)
		
		df_transcript = pd.DataFrame(chunk_list, columns = ['Text', 'Start', 'End'])
		
		df_transcript['Duration'] = df_transcript['End'] - df_transcript['Start']
		
		df_transcript = df_transcript[(df_transcript['Duration'] > 0)].copy()
		df_transcript = df_transcript.reset_index(drop=True)
		
		df_transcript['Length'] = df_transcript['Text'].str.split().str.len()
		
		df_long = df_transcript[(df_transcript['Length'] > 5)]
		df_shor = df_transcript[(df_transcript['Length'] <= 5)]
		
		long_seg = list(df_long.index.values)
		shor_seg = list(df_shor.index.values)	
		
		audiod = Audio(sample_rate=16000, mono='downmix')
		
		def compute_embedding(row):
			try:
				speaker = Segment(row['Start'], row['End'])
				waveform, sample_rate = audiod.crop('diarealsamples/' + conv_audio, speaker)
			except:
				speaker = Segment(row['Start'], row['End'] - 0.1)
				waveform, sample_rate = audiod.crop('diarealsamples/' + conv_audio, speaker)
				
			embedding = modelsp(waveform[None])
			return embedding
		
		embeddings = df_transcript.apply(compute_embedding, axis=1)
		
		dist_matrix = []
		for emb in embeddings:
			row = []
			for emb1 in embeddings:
				distance = cdist(emb, emb1, metric='cosine')[0][0]
				row.append(distance)
			dist_matrix.append(row)
		
		df_dist = pd.DataFrame(dist_matrix)
		df_dist = df_dist.fillna(2)
		
		dimensions = 50
		rows = len(df_transcript.index)
		
		utter = rows
		print(utter)
		if utter >= dimensions + 2:
			comp = dimensions
		else:
			comp = utter - 2
		print(comp)
		
		n_neighbors = rows // 4
		if n_neighbors < 2:
			n_neighbors = 2
		
		cluster_size = rows // 4
		if cluster_size < 3:
			cluster_size = 3
		
		clusterable_embedding = umap.UMAP(
		    n_neighbors=n_neighbors,
		    min_dist=.0,
		    n_components=3,
		    random_state=31416,
		    metric='cosine',
		).fit_transform(df_dist)
		
		clusterable_embedding_large = umap.UMAP(
		    n_neighbors=n_neighbors,
		    min_dist=.0,
		    n_components=comp,
		    random_state=31416,
		    metric='cosine',
		).fit_transform(df_dist)
		
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
		
		umap_df = pd.DataFrame(data = clusterable_embedding, columns = ['x', 'y', 'z'])
		umap_df['cluster_group'] = clustered
		
		df_umap = pd.concat([df_transcript, umap_df], axis=1)
		
		df_reassign = df_dist.copy()
		df_reassign['Labels'] = clustered
		
		for i in shor_seg:
			label_matrix = df_reassign.iloc[long_seg].groupby(['Labels'], as_index=False)[i].mean()
			new_label = int(label_matrix.loc[label_matrix[i].idxmin()]['Labels'])
			df_umap.at[i, 'cluster_group'] = new_label
		
		df_umap['cluster_group'] = le.fit_transform(df_umap['cluster_group'])
		df_umap['cluster_group'] = df_umap['cluster_group'] + 1
		
		clustered = df_umap['cluster_group'].tolist()
		
		print('Clusters: ' + ' '.join(str(i) for i in list(set(clustered))))
		print('Ratio of data points labelled: ' + str((np.sum(clustered) / rows).round(2)))
		print('Cluster quality (1 is perfect): ' + str((clusterer.relative_validity_).round(2)))
	
		###
		
		fig = px.scatter_3d(
		    df_umap, x='x', y='y', z='z',
		    color=df_umap['cluster_group'],
		    color_continuous_scale=px.colors.sequential.Jet,
		    hover_data=['Text', 'Duration', 'Length', 'cluster_group']
		)
		fig.update_scenes(xaxis_visible=False, yaxis_visible=False, zaxis_visible=False )
		fig.update_traces(marker_size = 4)
		fig.write_html('diarealsamples/' + base_name + '_hdbscan.html')
		
		###
		
		def secondsToStr(t):
		    return "%02d:%02d:%02d,%03d" % \
		        reduce(lambda ll,b : divmod(ll[0],b) + ll[1:],
		            [(round(t*1000),),1000,60,60])
		
		with open('diarealsamples/' + base_name + '.srt', 'w', encoding = 'utf-8') as f:
			for ind, col in df_umap.iterrows():
				ind = ind + 1
				f.write(str(ind) + '\n')
				f.write(secondsToStr(col['Start']) + ' --> ' + secondsToStr(col['End']) + '\n')
				#f.write('[SPEAKER ' + str(col['cluster_group']) + ' (' + col['Gender'] + ')]:' +  str(col['Text']) + '\n\n')
				f.write('[SPEAKER ' + str(col['cluster_group']) + ']:' +  str(col['Text']) + '\n\n')

