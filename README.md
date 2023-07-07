# Approaches to diarisation

_A testing repo to share code and thoughts on diarisation_

I am new to the field of NLP and struggling with diarisation. I have tried a number of state-of-the-art approaches. 

While several of them, namely those combining Whisper with Pyannote or NeMo, yield satisfactory results when it comes to the quality of the transcription and the aligment, speaker attribution is a different matter all together. 

Sometimes it works like a charm, sometimes it is a complete disaster, or anything in between. I wanted to understand why. During my tribulations, I came up with the approach showcased in the attached script.

This procedure (work in progress) yields much better results than anything else I had tried at the time of this writing. 

Note that these are just code snippets and not an installable module or library (when going through the code also keep in mind that I am chemist, not a developer).

Please, feel free to fork and contribute.

### REQUIREMENTS ###

SCRIPT 0: batch_diarize_stablets_hdbscan_novi4.py

This is the only script you need now. The others are there for just for reference.

It works with Python 3.8.17. Because of NeMo, it fails on Python 3.11.3. It it best to created a dedicated environment.

It expects to find the samples in mp3 format in the "samples" folder.

These are the dependencies explicitly imported:

```
scipy==1.10.1
plotly==5.15.0
pandas==1.5.3
numpy==1.22.4
pydub==0.25.1
sox==1.4.1
umap-learn==0.5.3
hdbscan==0.8.29
torch==2.0.1
stable-ts==2.6.4
scikit-learn==1.2.2
demucs @ git+https://github.com/facebookresearch/demucs@5d2ccf224f595b8654b0447e06f6adc866cca61a
nemo-toolkit @ git+https://github.com/NVIDIA/NeMo.git@c4e677a2d7aad47dbade8c3a0e47311a51d03bba
```
They may pull most of the others, but perhaps not all. Check the import errors.

In addition, the ffmpeg and sox libraries/executables need to be installed in your machine.

Hardware requirements: You will need at least 8 GB of RAM and a few GB of disk space for the models. 

This script neither requires nor probably will use any avaliable GPUs. You will need to hack it if you wish to take advantage of a GPU accelerator. 

### WORKFLOW ###

1. Audio pre-processing: voice isolation with [demucs](https://github.com/facebookresearch/demucs), plus conversion to 16-bit 16MHz mono WAV and normalisation with [pydub](https://github.com/jiaaro/pydub).
   
2. Transcription and timestamp synchronisation with [Whisper](https://github.com/openai/whisper) via [stable_ts](https://github.com/jianfch/stable-ts).

3. Custom post-processing of the stable_ts output in order to have more consistent sentence splitting and creation of SRT subtitles in "diarisamples".

4. Computes the embeddings for each segment with a [TitaNet](https://huggingface.co/nvidia/speakerverification_en_titanet_large) model. You will need [NeMo](https://github.com/NVIDIA/NeMo).

5. Computes all-versus-all cosine distance matrices from the TitaNet embeddings with scipy (this code chunk in particular is so ugly and inefficient that would make van Rossum cry).

6. Dimensionality reduction of the distance matrices with [UMAP](https://github.com/lmcinnes/umap). Even in cases where dimensionality reduction is not strictly required, I have observed that HDBSCAN works better with UMAP embeddings than it does on raw data. 

7. Clustering of the UMAP embeddings with [HDBSCAN](https://github.com/scikit-learn-contrib/hdbscan).
   
8. Reclustering. Short sentences are sometimes misattributed so we reassign them to the nearest cluster of long sentences. This works surprisingly well (better than HDBSCAN's approximate_predict).

9. Saves diarised SRT files along with interactive 3D HTML plots procuded with [plotly](https://github.com/plotly/plotly.py). Note that even though we are using only 3 dimensions for plotting, more (50 or as many as possible) are employed for the actual clustering.

### CALIBRATION ###

The 40 (22 public + 18 confidential) samples employed to develop this procedure contain 1-3 speakers and UMAP and HDBSCAN parametres have been calibrated accordingly, but, in principle, it should support any undetermined number of speakers. Heuristics or ML may be required in the future in order to guess the ideal parameter set for each sample (TODO). 
