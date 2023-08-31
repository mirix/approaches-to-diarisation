# Approaches to diarisation

_A testing repo to share code and thoughts on diarisation_

I am new to the field of NLP and currently working on diarisation (and beyond). I have tried a number of state-of-the-art approaches. 

While several of them, namely those combining Whisper with Pyannote or NeMo, typically yield satisfactory results when it comes to the quality of the transcription and the aligment, speaker attribution is a different matter all together. 

Sometimes it works like a charm, sometimes it is a complete disaster, or anything in between. I wanted to understand why. During my tribulations, I came up with the approach showcased in the attached script.

This procedure (work in progress) yields better results than anything else I had tried at the time of this writing. 

Note that these are just code snippets and not an installable module or library (when going through the code also keep in mind that I am chemist, not a developer). 

Please, feel free to fork and contribute.

### REQUIREMENTS ###

#### diarize_whisper_stablets_nemo_hdbscan.py ####

This is the only script you need now.

It works with Python 3.8. Because of NeMo, it fails with Python 3.11 (I have not tested any versions in between).

The installation instructions are now provided as a separate INSTALL file (tested and working).

The script expects to find the samples in mp3 format in the "samples" folder and saves the outputs to "diarealsamples".

Hardware requirements: You will need at least 16 GB of RAM or VRAM and a few GB of disk space for the models. 

This script can run on the CPU or the GPU (tested and working). You may need to hack it a bit though.

### WORKFLOW ###

1. Audio pre-processing: voice isolation with [demucs](https://github.com/facebookresearch/demucs), plus conversion to 16-bit 16MHz mono WAV and normalisation with [pydub](https://github.com/jiaaro/pydub).
   
2. Transcription and timestamp synchronisation with [Whisper](https://github.com/openai/whisper) via [stable_ts](https://github.com/jianfch/stable-ts). I have thoroughly tested several approaches and Stable Whisper still offers superior results.

3. Repunctuation and recapitalisation with a [NeMo model](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/nlp/punctuation_and_capitalization.html). This particular model abuses the period which results in excesive splitting. Short sentences are sometimes attributed to the wrong speaker. However, this prevents utterances from different speakers from being glued together. It is a trade-off. We have preferred having a few more short sentences in order to have more accurate speaker indentification at a later stage. If you are aware of a better puctuation model, please, let us know. We have tried many. Perhaps a multimodal one combining text and voice acitivity detection (VAD) would be ideal. However, in our limited experience, purely textual models seem to perform better than purely VAD ones.
   
4. Sentence splitting. In a nutshell, short sentences are split in punctuation marks such as period, exclamation and interrogation marks. If, however, a sentence grows longer than 44 words it will be split on the first comma. You can set your own maximal length, but know that it will have an influence on the clustering. Drastical changes may require recalibration.

5. Computation of the embeddings for each segment with a [TitaNet](https://huggingface.co/nvidia/speakerverification_en_titanet_large) model. You will need [NeMo](https://github.com/NVIDIA/NeMo).

6. Computation of all-versus-all cosine distance matrices from the TitaNet embeddings with scipy (this code chunk in particular is so ugly and inefficient that would make van Rossum cry).

7. Dimensionality reduction of the distance matrices with [UMAP](https://github.com/lmcinnes/umap). Even in cases where dimensionality reduction is not strictly required, we have observed that HDBSCAN seems to work better with UMAP embeddings than it does on raw data. 

8. Clustering of the UMAP embeddings with [HDBSCAN](https://github.com/scikit-learn-contrib/hdbscan). Only long sentences (> 5 words) are clustered.
   
9. Clustering of short sentences. Short sentences are assigned to the nearest cluster of long sentences. This is not perfect at the moment. The best solution is perhaps more accurate puctuation as described above.

10. The diarised SRT files along with interactive 3D HTML plots procuded with [plotly](https://github.com/plotly/plotly.py). Note that even though we are using only 3 dimensions for plotting, as many as possible up to 50 are employed for the actual clustering.

### CALIBRATION ###

The 36 samples employed to develop this procedure contain 1-3 speakers and UMAP/HDBSCAN parametres have been calibrated accordingly. In principle, this pipeline should support any undetermined number of speakers. Heuristics or ML may be required in the future in order to guess the ideal parameter set for each sample. 
