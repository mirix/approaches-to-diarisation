# Approaches to diarisation

_A testing repo to share code and thoughts on diarisation_

I am new to the field of NLP and currently working on diarisation (and beyond). I have tried a number of state-of-the-art approaches. 

While several of them, namely those combining Whisper with Pyannote or NeMo, typically yield satisfactory results when it comes to the quality of the transcription and the aligment, speaker attribution is a different matter all together. 

Sometimes it works like a charm, sometimes it is a complete disaster, or anything in between. I wanted to understand why. During my tribulations, I came up with the approach showcased in the attached script.

This procedure (work in progress) yields much better results than anything else I had tried at the time of this writing. 

Note that these are just code snippets and not an installable module or library (when going through the code also keep in mind that I am chemist, not a developer). 

Please, feel free to fork and contribute.

### REQUIREMENTS ###

#### diarize_whisper_stablets_hdbscan.py ####

This is the only script you need now.

It works with Python 3.8. Because of NeMo, it fails with Python 3.11 (I have not tested any versions in between).

The installation instructions are now provided as a separate INSTALL file (tested and working).

The script expects to find the samples in mp3 format in the "samples" folder and saves the outputs to "diarealsamples".

Hardware requirements: You will need at least 16 GB of RAM and a few GB of disk space for the models. 

This script can run on the CPU and does not require a GPU accelerator. It will use it if present though (tested and working), but you will need to hack the script if you wish to take full advantage of multiple GPUs.

### WORKFLOW ###

1. Audio pre-processing: voice isolation with [demucs](https://github.com/facebookresearch/demucs), plus conversion to 16-bit 16MHz mono WAV and normalisation with [pydub](https://github.com/jiaaro/pydub).
   
2. Transcription and timestamp synchronisation with [Whisper](https://github.com/openai/whisper) via [stable_ts](https://github.com/jianfch/stable-ts).

3. Custom post-processing of the stable_ts output in order to have more consistent sentence splitting.

4. Computes the embeddings for each segment with a [TitaNet](https://huggingface.co/nvidia/speakerverification_en_titanet_large) model. You will need [NeMo](https://github.com/NVIDIA/NeMo).

5. Computes all-versus-all cosine distance matrices from the TitaNet embeddings with scipy (this code chunk in particular is so ugly and inefficient that would make van Rossum cry).

6. Dimensionality reduction of the distance matrices with [UMAP](https://github.com/lmcinnes/umap). Even in cases where dimensionality reduction is not strictly required, I have observed that HDBSCAN seems to work better with UMAP embeddings than it does on raw data. 

7. Clustering of the UMAP embeddings with [HDBSCAN](https://github.com/scikit-learn-contrib/hdbscan).
   
8. Reclustering. Short sentences are sometimes misattributed so we reassign them to the nearest cluster of long sentences. This works surprisingly well (better than HDBSCAN's approximate_predict).

9. Saves the diarised SRT files along with interactive 3D HTML plots procuded with [plotly](https://github.com/plotly/plotly.py). Note that even though we are using only 3 dimensions for plotting, as many as possible up to 50 are employed for the actual clustering.

### CALIBRATION ###

The 40 (22 public + 18 confidential) samples employed to develop this procedure contain 1-3 speakers and UMAP and HDBSCAN parametres have been calibrated accordingly, but, in principle, it should support any undetermined number of speakers. Heuristics or ML may be required in the future in order to guess the ideal parameter set for each sample (ToDo). 
