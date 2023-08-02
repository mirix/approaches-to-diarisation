I have added a new script that computes sentence-wise  [VAD/PAD](https://en.wikipedia.org/wiki/PAD_emotional_state_model) values using the [audEERING model](https://huggingface.co/audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim). 

It uses as inputs the WAV and SRT files produced by the diarisation script located in the upper level folder. 

This script (test_emotions.py) works with Python 3.11 and possibly previous versions and the requirements should be straightforward to install.

It then attempts to translate the VAD values into [Ekman basic emotions](https://www.paulekman.com/universal-emotions/) and produces a new set of SRT files annotated with the predominant emotion.

The conversion from VAD into Ekman and the selection of the predominant emotion have not been properly calibrated yet but, in my humble opinion, the procedure already yields useful enough results.

Namely, a conversation segment rich in negatively-flagged sentences typically deserves further inspection.

Each emotion paradigm has a different scope of applicability and I therefore though it would be useful being able to translate from one into another.

This work is loosely inspired by:

https://doi.org/10.3390/electronics10212643
