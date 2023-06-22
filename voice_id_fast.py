import torch

from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
from pyannote.audio import Audio
from pyannote.core import Segment

from faster_whisper import WhisperModel

from scipy.spatial.distance import cdist
from functools import reduce
import pandas as pd

import os
n_cores = '6'
os.environ['OMP_NUM_THREADS'] = n_cores
os.environ['MKL_NUM_THREADS'] = n_cores
 
audio_file = 'DxxAwDHgQhE.wav'
#batch_size = 32
device = 'cpu'
compute_type = 'float32'
model_size = 'large-v2'
voice_thres = .95

model = WhisperModel(model_size, device=device, compute_type=compute_type)
segments, info = model.transcribe(audio_file, beam_size=5, word_timestamps=True)

print("Detected language '%s' with probability %f" % (info.language, info.language_probability))

chunk_list = []
for segment in segments:
	tup = (segment.text, segment.start, segment.end)
	chunk_list.append(tup)

df_transcript = pd.DataFrame(chunk_list, columns = ['Text', 'Start', 'End'])

df_transcript['Duration'] = df_transcript['End'] - df_transcript['Start']
df_transcript['Length'] = df_transcript['Text'].str.split().str.len()

df_transcript = df_transcript.sort_values(by=['Length', 'Duration', 'Start'], ascending=[False, False, True])
df_transcript.reset_index(drop=True, inplace=True)

print(df_transcript)

model = PretrainedSpeakerEmbedding('speechbrain/spkrec-ecapa-voxceleb', device=torch.device('cpu'))

audio = Audio(sample_rate=16000, mono='downmix')

speaker1 = Segment(df_transcript.at[0, 'Start'], df_transcript.at[0, 'End'])
waveform1, sample_rate = audio.crop(audio_file, speaker1)
embedding1 = model(waveform1[None])

def voice_distance(start, end):
	speaker2 = Segment(start, end)
	waveform2, sample_rate = audio.crop(audio_file, speaker2)
	embedding2 = model(waveform2[None])
	distance = cdist(embedding1, embedding2, metric='cosine')
	return distance[0][0]

df_transcript['SPEAKER 1'] = df_transcript.apply(lambda x: voice_distance(x['Start'], x['End']), axis=1)

is_speaker2 = len(df_transcript[df_transcript['SPEAKER 1'].gt(voice_thres)])

if is_speaker2 > 0:
	speaker2 = df_transcript[df_transcript['SPEAKER 1'].gt(voice_thres)].index[0]
	speaker1 = Segment(df_transcript.at[speaker2, 'Start'], df_transcript.at[speaker2, 'End'])
	waveform1, sample_rate = audio.crop(audio_file, speaker1)
	embedding1 = model(waveform1[None])
	df_transcript['SPEAKER 2'] = df_transcript.apply(lambda x: voice_distance(x['Start'], x['End']), axis=1)
	is_speaker3 = len(df_transcript[(df_transcript['SPEAKER 1'].gt(voice_thres)) & (df_transcript['SPEAKER 2'].gt(voice_thres))])
	if is_speaker3 > 0:
		speaker3 = df_transcript[(df_transcript['SPEAKER 1'].gt(voice_thres)) & (df_transcript['SPEAKER 2'].gt(voice_thres))].index[0]
		speaker1 = Segment(df_transcript.at[speaker3, 'Start'], df_transcript.at[speaker3, 'End'])
		waveform1, sample_rate = audio.crop(audio_file, speaker1)
		embedding1 = model(waveform1[None])
		df_transcript['SPEAKER 3'] = df_transcript.apply(lambda x: voice_distance(x['Start'], x['End']), axis=1)

df_transcript['Speaker'] = df_transcript[[col for col in df_transcript.columns if 'SPEAKER' in col]].idxmin(axis=1)

df_transcript = df_transcript.sort_values(by=['Start'], ascending=[True])
df_transcript.reset_index(drop=True, inplace=True)

print(df_transcript)

df_transcript.to_csv('transcript_test.csv', encoding='utf-8', index=False)

def secondsToStr(t):
    return "%02d:%02d:%02d,%03d" % \
        reduce(lambda ll,b : divmod(ll[0],b) + ll[1:],
            [(round(t*1000),),1000,60,60])

with open(audio_file.rsplit('.', 1)[0] + '.srt', 'w', encoding = 'utf-8') as f:
	for ind, col in df_transcript.iterrows():
		ind = ind + 1
		f.write(str(ind) + '\n')
		f.write(secondsToStr(col['Start']) + ' --> ' + secondsToStr(col['End']) + '\n')
		f.write('[' + col['Speaker'] + ']: ' + col['Text'] + '\n\n')
