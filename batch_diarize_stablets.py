import os
n_cores = '6'
os.environ['OMP_NUM_THREADS'] = n_cores
os.environ['MKL_NUM_THREADS'] = n_cores

import yt_dlp
import ffmpeg
import shutil
from urllib import request
import demucs.separate
from pydub import AudioSegment, effects  
import shlex

import stable_whisper

from functools import reduce
import pandas as pd

from scipy.io.wavfile import read as wavread
from scipy.io.wavfile import write as wavwrite
import numpy as np
 
model_size = 'large-v2'
modelw = stable_whisper.load_model(model_size)

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
		.output(conv_audio, **{'acodec': 'pcm_s16le', 'ac': 1, 'ar': 16000})
		.run())
		rawsound = AudioSegment.from_file(conv_audio, 'wav')  
		normalizedsound = effects.normalize(rawsound)  
		normalizedsound.export(conv_audio, format='wav')
		samplerate, data = wavread(conv_audio)
		wavwrite(conv_audio, samplerate, data.astype(np.int16))
		
### TRANSCRIBE ###

stops = ('。', '．', '.', '！', '!', '?', '？')
abbre = ('Dr.', 'Mr.', 'Mrs.', 'Ms.', 'vs.', 'Prof.')

for wavs in os.scandir('diarisamples'):
	
	if wavs.is_file() and wavs.path.endswith('.wav'):
		
		base_name = wavs.name[:-4]
		print(base_name)
		conv_audio = 'diarisamples/' + base_name + '.wav'
		
		result = modelw.transcribe(conv_audio, regroup='sp=.* /。/?/？/．/!/！')
		results = result.to_dict()['segments']
		
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
		
		def secondsToStr(t):
		    return "%02d:%02d:%02d,%03d" % \
		        reduce(lambda ll,b : divmod(ll[0],b) + ll[1:],
		            [(round(t*1000),),1000,60,60])
		
		with open(conv_audio.rsplit('.', 1)[0] + '_nodiarisation.srt', 'w', encoding = 'utf-8') as f:
			for ind, col in df_transcript.iterrows():
				ind = ind + 1
				f.write(str(ind) + '\n')
				f.write(secondsToStr(col['Start']) + ' --> ' + secondsToStr(col['End']) + '\n')
				f.write(str(col['Text']) + '\n\n')

shutil.rmtree('tmp', ignore_errors=True)


