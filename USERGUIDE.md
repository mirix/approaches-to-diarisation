
1.    Clone this repo and follow the [installation instructions](https://github.com/mirix/approaches-to-diarisation/blob/main/INSTALL.md):
  

2.    Check and edit the most recent script (GPU):

`diarize_whisper_stablets_nemo_hdbscan_rapids.py`

Namely, you need to adapt the GPU configuration to the number of GPUs you have (top of the script).

(if you don't have a GPU or do not want to use it, hack the other script accordingly)


3.    The script expects to find the recordings in MP3 format in a subfolder called samples.

Check the PREPROCESS AUDIO section if you wish to change the input folder or the input format.


4.    Run the script:

`python diarize_whisper_stablets_nemo_hdbscan_rapids.py`

If there are any error messages read them carefully and try to figure out what is missing.
