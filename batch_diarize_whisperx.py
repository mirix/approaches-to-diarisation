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

import whisperx

from functools import reduce
import pandas as pd

from scipy.io.wavfile import read as wavread
from scipy.io.wavfile import write as wavwrite
import numpy as np

#n_speakers = 3
#batch_size = 64 
device = 'cpu'
compute_type = 'float32'
model_size = 'large-v2'

modelw = whisperx.load_model(model_size, device, compute_type=compute_type)

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
		
		# 1. Transcribe with original whisper (batched)
		model = whisperx.load_model('large-v2', device, compute_type=compute_type)
		
		audio = whisperx.load_audio(conv_audio)
		result = model.transcribe(audio)
		
		# 2. Align whisper output
		model_a, metadata = whisperx.load_align_model(language_code=result['language'], device=device, model_name='WAV2VEC2_ASR_LARGE_LV60K_960H')
		result = whisperx.align(result['segments'], model_a, metadata, audio, device, return_char_alignments=False)
		
		# 3. Assign speaker labels
		diarize_model = whisperx.DiarizationPipeline(use_auth_token='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', device=device)
		
		#diarize_segments = diarize_model(audio_file)
		diarize_segments = diarize_model(conv_audio, min_speakers=1, max_speakers=3)
		
		result = whisperx.assign_word_speakers(diarize_segments, result)
		
		def secondsToStr(t):
		    return "%02d:%02d:%02d,%03d" % \
		        reduce(lambda ll,b : divmod(ll[0],b) + ll[1:],
		            [(round(t*1000),),1000,60,60])
		
		with open(conv_audio.split('.')[0] + '.srt', 'w', encoding = 'utf-8') as f:
			ind = 1
			for sentence in result['segments']:
				f.write(str(ind) + '\n')
				f.write(secondsToStr(sentence['start']) + ' --> ' + secondsToStr(sentence['end']) + '\n')
				f.write('[' + str(sentence['speaker']) + ']: ' + str(sentence['text']) + '\n\n')
				ind = ind + 1

shutil.rmtree('tmp', ignore_errors=True)


