import os
n_cores = '6'
os.environ['OMP_NUM_THREADS'] = n_cores
os.environ['MKL_NUM_THREADS'] = n_cores

import yt_dlp
import ffmpeg
import shutil
from urllib import request
import demucs.separate
import shlex

import torch

from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
from pyannote.audio import Audio
from pyannote.core import Segment

from faster_whisper import WhisperModel
#import stable_whisper
#import whisperx

from scipy.spatial.distance import cdist
from functools import reduce
import pandas as pd

from scipy.io.wavfile import read as wavread
from scipy.io.wavfile import write as wavwrite
import numpy as np

from decimal import Decimal as D, ROUND_DOWN

def truncate(f):
	return D(f).quantize(D('0.01'), rounding=ROUND_DOWN)

#n_speakers = 3
batch_size = 64 
device = 'cpu'
compute_type = 'float32'
model_size = 'large-v2'
voice_thres = .85


#modelw = whisperx.load_model(model_size, device, compute_type=compute_type)

modelw = WhisperModel(model_size, device=device, compute_type=compute_type)
modelsp = PretrainedSpeakerEmbedding('speechbrain/spkrec-ecapa-voxceleb', device=torch.device('cpu'))
#modelsp = PretrainedSpeakerEmbedding('nvidia/speakerverification_en_titanet_large', device=torch.device('cpu'))

if not os.path.isdir('diarisamples'):
	os.makedirs('diarisamples')

if not os.path.isdir('tmp'):
	os.makedirs('tmp')

### DOWNLOAD ###
	
base_url = 'https://www.youtube.com/watch?v='
yt_ids = ['DxxAwDHgQhE', 'Fyb2AiF1feI', 'qHrN5Mf5sgo']

urls = [base_url + s for s in yt_ids]

ydl_opts = {
    'paths': {'home': 'tmp'},
    'outtmpl': {'default': '%(id)s.%(ext)s'},
    'format': 'm4a/bestaudio/best',
    'postprocessors': [{
        'key': 'FFmpegExtractAudio',
        'preferredcodec': 'm4a',
    }]
}

for yid in yt_ids:
	conv_audio = 'diarisamples/' + yid + '.wav'
	if not os.path.exists(conv_audio) or os.path.getsize(conv_audio) < 100000:
		url = base_url + yid
		with yt_dlp.YoutubeDL(ydl_opts) as ydl:
			error_code = ydl.download(url)

conv_audios = ['https://callhounds.com/wp-content/uploads/2020/10/Travel-Reservation.mp3',
				'https://callhounds.com/wp-content/uploads/2020/10/Real-State-Lead-Gen-1.mp3',
				'https://callhounds.com/wp-content/uploads/2020/10/Real-State-Lead-Gen-4.mp3',
				'https://callhounds.com/wp-content/uploads/2020/10/Health-Insurance-1.mp3',
				'https://callhounds.com/wp-content/uploads/2020/10/Health-Insurance-2.mp3',
				'https://callhounds.com/wp-content/uploads/2020/10/Technician-Recruitment.mp3',
				'https://callhounds.com/wp-content/uploads/2020/10/Logistics.mp3',
				'https://callhounds.com/wp-content/uploads/2022/12/HP-Technician-Dispat-1.wav',
				'https://callhounds.com/wp-content/uploads/2022/12/HP-Technician-Dispatch-2.wav',
				'https://callhounds.com/wp-content/uploads/2022/12/Technician-Assignment.wav',
				'https://callhounds.com/wp-content/uploads/2020/10/Architecture-Firm.mp3',
				'https://callhounds.com/wp-content/uploads/2020/10/Costco.mp3',
				'https://callhounds.com/wp-content/uploads/2020/10/Custom-Home-Builder.mp3',
				'https://callhounds.com/wp-content/uploads/2020/10/Fundraising.mp3',
				'https://callhounds.com/wp-content/uploads/2020/10/Local-Plumber.mp3',
				'https://callhounds.com/wp-content/uploads/2020/10/Property-Management-Office.mp3',
				'https://callhounds.com/wp-content/uploads/2020/10/Security-System-Monitoring.mp3',
				'https://callhounds.com/wp-content/uploads/2020/10/Third-Party-Verification.mp3',
				'https://callhounds.com/wp-content/uploads/2020/10/Travel.mp3'
				]

for url in conv_audios:
	name = url.split('/')[-1].replace('-', '_')
	base_name = name.split('.')[0]
	conv_audio = 'diarisamples/' + base_name + '.wav'
	if not os.path.exists(conv_audio) or os.path.getsize(conv_audio) < 100000:
		request.urlretrieve(url, 'tmp/' + name)

### CONVERT ###

ext = ('.m4a', '.mp3', '.wav')

for audios in os.scandir('tmp'):
	name = audios.name
	base_name = name[:-4]
	conv_audio = 'diarisamples/' + base_name + '.wav'
	if (audios.is_file() and audios.path.endswith(ext)) and (not os.path.exists(conv_audio) or os.path.getsize(conv_audio) < 100000):
		demucs.separate.main(shlex.split('--two-stems vocals -n mdx_extra ' + 'tmp/' + name + ' -o tmp'))
		(ffmpeg
		.input('tmp/mdx_extra/' + base_name + '/vocals.wav')
		.output(conv_audio, **{'acodec': 'pcm_s16le', 'ac': 1, 'ar': 16000, 'af': "silenceremove=start_periods=1:start_duration=1:start_threshold=-60dB:detection=peak,aformat=dblp,areverse,silenceremove=start_periods=1:start_duration=1:start_threshold=-60dB:detection=peak,aformat=dblp,areverse"})
		#.output(conv_audio, **{'acodec': 'pcm_s16le', 'ac': 1, 'ar': 16000})
		.run())
		samplerate, data = wavread(conv_audio)
		wavwrite(conv_audio, samplerate, data.astype(np.int16))
		
### TRANSCRIBE ###

for wavs in os.scandir('diarisamples'):
	
	if wavs.is_file() and wavs.path.endswith('.wav'):
		
		base_name = wavs.name[:-4]
		print(base_name)
		conv_audio = 'diarisamples/' + base_name + '.wav'
		# and not os.path.exists('diarisamples/' + base_name + '.srt'):	
		
		#audiow = whisperx.load_audio(conv_audio)
		#result = modelw.transcribe(audiow, batch_size=batch_size)
		#model_a, metadata = whisperx.load_align_model(language_code=result['language'], device=device, model_name='WAV2VEC2_ASR_LARGE_LV60K_960H')
		#result = whisperx.align(result['segments'], model_a, metadata, audiow, device, return_char_alignments=False)
		
		segments, info = modelw.transcribe(conv_audio, beam_size=5, word_timestamps=True)
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
		
		audiod = Audio(sample_rate=16000, mono='downmix')
		
		speaker1 = Segment(df_transcript.at[0, 'Start'], df_transcript.at[0, 'End'])
		waveform1, sample_rate = audiod.crop(conv_audio, speaker1)
		embedding1 = modelsp(waveform1[None])
		
		def voice_distance(start, end):
			end = truncate(end)
			speaker2 = Segment(start, end)
			waveform2, sample_rate = audiod.crop(conv_audio, speaker2)
			embedding2 = modelsp(waveform2[None])
			distance = cdist(embedding1, embedding2, metric='cosine')
			return distance[0][0]
		
		df_transcript['SPEAKER 1'] = df_transcript.apply(lambda x: voice_distance(x['Start'], x['End']), axis=1)
		
		is_speaker2 = len(df_transcript[df_transcript['SPEAKER 1'].gt(voice_thres)])
		
		if is_speaker2 > 0:
			speaker2 = df_transcript[df_transcript['SPEAKER 1'].gt(voice_thres)].index[0]
			speaker1 = Segment(df_transcript.at[speaker2, 'Start'], df_transcript.at[speaker2, 'End'])
			waveform1, sample_rate = audiod.crop(conv_audio, speaker1)
			embedding1 = modelsp(waveform1[None])
			df_transcript['SPEAKER 2'] = df_transcript.apply(lambda x: voice_distance(x['Start'], x['End']), axis=1)
			is_speaker3 = len(df_transcript[(df_transcript['SPEAKER 1'].gt(voice_thres)) & (df_transcript['SPEAKER 2'].gt(voice_thres))])
			if is_speaker3 > 0:
				speaker3 = df_transcript[(df_transcript['SPEAKER 1'].gt(voice_thres)) & (df_transcript['SPEAKER 2'].gt(voice_thres))].index[0]
				speaker1 = Segment(df_transcript.at[speaker3, 'Start'], df_transcript.at[speaker3, 'End'])
				waveform1, sample_rate = audiod.crop(conv_audio, speaker1)
				embedding1 = modelsp(waveform1[None])
				df_transcript['SPEAKER 3'] = df_transcript.apply(lambda x: voice_distance(x['Start'], x['End']), axis=1)
		
		df_transcript['Speaker'] = df_transcript[[col for col in df_transcript.columns if 'SPEAKER' in col]].idxmin(axis=1)
		
		df_transcript = df_transcript.sort_values(by=['Start'], ascending=[True])
		df_transcript.reset_index(drop=True, inplace=True)
		
		#df_transcript.to_csv('transcript_test.csv', encoding='utf-8', index=False)
		
		def secondsToStr(t):
		    return "%02d:%02d:%02d,%03d" % \
		        reduce(lambda ll,b : divmod(ll[0],b) + ll[1:],
		            [(round(t*1000),),1000,60,60])
		
		with open(conv_audio.rsplit('.', 1)[0] + '.srt', 'w', encoding = 'utf-8') as f:
			for ind, col in df_transcript.iterrows():
				ind = ind + 1
				f.write(str(ind) + '\n')
				f.write(secondsToStr(col['Start']) + ' --> ' + secondsToStr(col['End']) + '\n')
				f.write('[' + str(col['Speaker']) + ']: ' + str(col['Text']) + '\n\n')

shutil.rmtree('tmp', ignore_errors=True)


