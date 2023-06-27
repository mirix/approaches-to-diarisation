# Approaches to diarisation

_A testing repo to share code and thoughts on diarisation_

I am new to the field of NLP and struggling with diarisation.

I have tried a number of state-of-the-art approaches. 

While several of them, namely those combining Whisper with Pyannote or Nemo, yield satisfactory results when it comes to the quality of the transcription and the aligment, speaker attribution is a different matter all together. 

Sometimes it works like a charm, sometimes it is a complete disaster. I wanted to understand why. During my tribulations, I came up with the approach showcased in the attached script (I am chemist, not a developer, please, keep that in mind).

The approach is simple, it selects the longest segment (number of words seems to work better than characters or duration), compares it to all other segments (currenty using cosine distance from ECAPA embeddings, which seem to work better than TitaNet in this particular scenario) and sets a similarity threshold to determine if there is more than one speaker. If there is, it takes the longest sentence meeting the threshold criterion, rinses and repeats. Right now it goes up to three possible speakers, but it could be looped. In the end, one ends up with one column of cosine distances per speaker. The attribution is made according to the shortest distance per row. 

I have tested in over a dozen samples and this hypersimplistic approach seems to work better than anything I had tried before.

The quality of speaker attribution depends a lot on the initial sentences chosen as reference for each speaker (namely the first one) as well as, to a lesser extent, to the distance threshold. I believe that, with a bit of heuristics in order to adapt those two variables for each sample track, this could possibly constitute a reliable approach.

It currently relies on fast_whisper, pyannote and SpeechBrain embeddings. It was tested on Manjaro Linux (Python 3.11.3) with the latest pip versions of all the requirements. The default device is cpu and the version of pythorch is also cpu-only. 

If a real developer or NLP enthusiast would be willing to have a lot at this, perhaps we could get to something.

It processes 22 samples and store the processed wav files along with the diarised subtitled in srt format in a folder called "diarisamples". 

The script batch_diarize_whisperx.py is the standard Whisper X procedure whereas batch_diarize_fasterwhisper.py is the very first prototype of my procedure.

My procedure produces better results. 
