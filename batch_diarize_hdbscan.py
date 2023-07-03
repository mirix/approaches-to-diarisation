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

#modelsp = PretrainedSpeakerEmbedding('speechbrain/spkrec-ecapa-voxceleb', device=torch.device('cpu'))
modelsp = PretrainedSpeakerEmbedding('nvidia/speakerverification_en_titanet_large', device=torch.device('cpu'))

for wavs in os.scandir('diarisamples'):
	if wavs.is_file() and wavs.path.endswith('.wav'):
		
		base_name = wavs.name[:-4]
		print(' ')
		print(base_name)
		conv_audio = base_name + '.wav'
		sub_file = base_name + '_nodiarisation.srt'
		
		srt_file = open('diarisamples/' + sub_file, 'r')
		
		chunk_list = []
		for sub in srt.parse(srt_file):
			index = int(sub.index)
			start = float(str(sub.start.seconds) + '.' + str(sub.start.microseconds))
			end = float(str(sub.end.seconds) + '.' + str(sub.end.microseconds))
			#text = re.sub('\[.*?\]', '', sub.content.replace(']:  ', ']'))
			text = sub.content
			row = (index, text, start, end)
			chunk_list.append(row)
		
		df_transcript = pd.DataFrame(chunk_list, columns = ['Index', 'Text', 'Start', 'End'])
		
		df_transcript['Duration'] = df_transcript['End'] - df_transcript['Start']
		df_transcript['Length'] = df_transcript['Text'].str.split().str.len()
		
		audiod = Audio(sample_rate=16000, mono='downmix')
		
		def compute_embedding(row):
			speaker = Segment(row['Start'], row['End'])
			waveform, sample_rate = audiod.crop('diarisamples/' + conv_audio, speaker)
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
		
		#print(df_dist)
		#df_dist.to_csv('test_srt.csv', index=False)
		
		###
		utter = df_transcript.shape[0]
		print(utter)
		if utter >= 12:
			comp = 10
		else:
			comp = utter - 2
		print(comp)
		
		n_neighbors = df_transcript.shape[0] // 4
		if n_neighbors < 2:
			n_neighbors = 2
		#n_neighbors = 20
		
		cluster_size = df_transcript.shape[0] // 4
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
		    #metric='manhattan',
		    #metric='canberra',
		    metric='cosine',
		    #metric='euclidean',
		    #metric='correlation',
		).fit_transform(df_dist)
		
		clf = hdbscan.HDBSCAN(
		    min_samples=1,
		    min_cluster_size=cluster_size,
		    #cluster_selection_method='eom',
		    cluster_selection_method='leaf',
		    cluster_selection_epsilon=1.1,
		    gen_min_span_tree=True
		)
		
		if comp >= 10:
			emb_matrix = clusterable_embedding_large
		else:
			emb_matrix = df_dist
		
		labels = clf.fit_predict(emb_matrix)
		
		#clustered = (labels >= 0)
		clustered = labels
		
		#print(clusterable_embedding)
		#print(labels)
		print('Clusters: ' + ' '.join(str(i) for i in list(set(labels.flat))))
		print('Ratio of data points labelled: ' + str((np.sum(clustered) / df_transcript.shape[0]).round(2)))
		print('Cluster quality (1 is perfect): ' + str((clf.relative_validity_).round(2)))
		
		
		umap_df = pd.DataFrame(data = clusterable_embedding, columns = ['x', 'y', 'z'])
		umap_df['cluster_group'] = labels
		umap_df['cluster_group'] = umap_df['cluster_group'] + 1
		
		df_umap = pd.concat([df_transcript, umap_df], axis=1)
		
		###
		
		fig = px.scatter_3d(
		    df_umap, x='x', y='y', z='z',
		    color=df_umap['cluster_group'],
		    #color_discrete_sequence=px.colors.qualitative.Dark24,
		    color_continuous_scale=px.colors.sequential.Jet,
		    hover_data=['Index', 'Text', 'Duration', 'Length', 'cluster_group']
		)
		fig.update_scenes(xaxis_visible=False, yaxis_visible=False, zaxis_visible=False )
		fig.update_traces(marker_size = 4)
		#fig.show()
		fig.write_html('diarisamples/' + base_name + '_hdbscan.html')
		
		###
		
		def secondsToStr(t):
		    return "%02d:%02d:%02d,%03d" % \
		        reduce(lambda ll,b : divmod(ll[0],b) + ll[1:],
		            [(round(t*1000),),1000,60,60])
		
		with open('diarisamples/' + base_name + '.srt', 'w', encoding = 'utf-8') as f:
			for ind, col in df_umap.iterrows():
				ind = ind + 1
				f.write(str(ind) + '\n')
				f.write(secondsToStr(col['Start']) + ' --> ' + secondsToStr(col['End']) + '\n')
				f.write('[SPEAKER ' + str(col['cluster_group']) + ']:' +  str(col['Text']) + '\n\n')

