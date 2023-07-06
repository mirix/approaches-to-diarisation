# Approaches to diarisation

_A testing repo to share code and thoughts on diarisation_

I am new to the field of NLP and struggling with diarisation. I have tried a number of state-of-the-art approaches. 

While several of them, namely those combining Whisper with Pyannote or NeMo, yield satisfactory results when it comes to the quality of the transcription and the aligment, speaker attribution is a different matter all together. 

Sometimes it works like a charm, sometimes it is a complete disaster, or anything in between. I wanted to understand why. During my tribulations, I came up with the approach showcased in the attached scripts (when going through my code keep in mind that I am chemist, not a developer).

Note that these are just code snippets and not an installable module or library.


SCRIPT 1: batch_diarize_stablets.py

0. If required, downloads the sample audio files from different sources and saves them in a folder called "diarisamples". 

1. Audio pre-processing: voice isolation with [demucs](https://github.com/facebookresearch/demucs), conversion to 16-bit 16MHz WAV with ffmpeg, normalisation with [pydub](https://github.com/jiaaro/pydub), and resaving with scipy (this step is required to prevent out of range errors from [pyannote](https://github.com/pyannote/pyannote-audio) segment timestamps).
   
2. Transcription and timestamp synchronisation with [Whisper](https://github.com/openai/whisper) via [stable_ts](https://github.com/jianfch/stable-ts).

3. Post-processing of the stable_ts output in order to have more consistent sentence splitting and creation of SRT subtitles in "diarisamples".

Script 1 was tested with Python 3.11.3:

```
yt-dlp==2023.6.22
urllib3==2.0.3
ffmpeg-python==0.2.0
pydub==0.25.1
scipy==1.11.0
numpy==1.24.4
pandas==2.0.2
stable-ts==2.6.4
demucs @ git+https://github.com/facebookresearch/demucs@5d2ccf224f595b8654b0447e06f6adc866cca61a
```
Note that, if you clone this repo, you do not need to run Script 1 at all.


SCRIPT 2: batch_diarize_hdbscan_new.py

0. Takes the SRT and WAV files generated by the previous script from "diarisamples" as input.

1. Computes the embeddings for each segment with a [TitaNet](https://huggingface.co/nvidia/speakerverification_en_titanet_large) model. Alternatively, [ECAPA-TDNN](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb) can be used, but it yields somehow less accurate results. Note that in the first case you will need [NeMo](https://github.com/NVIDIA/NeMo) which is not compatible with the most recent Python.

2. Computes all-versus-all cosine distance matrices from the TitaNet embeddings with scipy (this code chunk in particular is so ugly and inefficient that would make van Rossum cry).

3. Dimensionality reduction of the distance matrices with [UMAP](https://github.com/lmcinnes/umap). Even in cases where dimensionality reduction is not strictly required, I have observed that HDBSCAN works better with UMAP embeddings than it does on raw data. 

4. Clustering of the UMAP embeddings with [HDBSCAN](https://github.com/scikit-learn-contrib/hdbscan).
   
5. The new script reclusters short sentences using the clusters of long sentences as reference. This works surprisingly well (better than HDBSCAN's approximate_predict).

6. Interactive 3D plots with [plotly](https://github.com/plotly/plotly.py) and saves the diarised SRT files. Note that even though we are using only 3 dimensions for plotting, more (50 or as many as possible) are employed for the actual clustering.

Script 2 was tested on Python 3.8.16:

```
scikit-learn==1.2.2
umap-learn==0.5.3
torch==2.0.1
torchaudio==2.0.2
scipy==1.10.1
pydub==0.25.1
srt==3.5.3
pandas==1.5.3
numpy==1.22.4
hdbscan==0.8.29
plotly==5.15.0
pyannote.audio @ git+https://github.com/pyannote/pyannote-audio@11b56a137a578db9335efc00298f6ec1932e6317
nemo-toolkit @ git+https://github.com/NVIDIA/NeMo.git@c4e677a2d7aad47dbade8c3a0e47311a51d03bba
```

If you use ECAPA-TDNN instead of TitaNet, you will not need NeMo and the script should also work with a more recent Python version.

The 22 samples used contain 1-3 speakers and UMAP and HDBSCAN parametres have been calibrated accordingly, but, in principle, this procedure should support any undetermined number of speakers. Heuristics or ML may be required in the future in order to guess the ideal parameter set for each sample (TODO). 

This procedure (work in progress) yields better results than anything else I had tried at the time of this writing. Please, feel free to fork and contribute.
