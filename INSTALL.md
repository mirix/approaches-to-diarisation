CREATE VIRTUAL ENVIRONMENT

For instance, on Arch Linux or derivatives:

1. Install yay (optional):

$ sudo pacman -S yay

2. Install Python 3.8:

$ yay python38

3. Create a virtual environment with Python 3.8 (I called it nemo):

$ cd ~/Downloads
$ python3.8 -m venv nemo

4. Activate the new environment:

$ source ~/Downloads/nemo/bin/activate

For verification purposes, outside of the environment:
$ python --version
Python 3.11.3

Inside of the environment:
Python 3.8.17
(what matters is just the major and the minor, 3.8, the patch, 17, is irrelevant)

CLONE REPO AND INSTALL THE REQUIREMENTS

5. Clone this repo:

git clone https://github.com/mirix/approaches-to-diarisation.git
cd approaches-to-diarisation

6. Install the requirements

pip install -r requirements.txt

7. Install demucs

python -m pip install -U "git+https://github.com/facebookresearch/demucs#egg=demucs"

8. Install Nvidia NeMo

$ sudo pacman -S libsndfile ffmpeg
$ pip install cython
$ python -m pip install "git+https://github.com/NVIDIA/NeMo.git@main#egg=nemo_toolkit[asr]"

9. Reinstall Pytorch (OPTINAL, only if you want a version that is different from the one pulled by NeMo):

cpu (nightly):

$ pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cpu

gpu (nightly for CUDA 12.1):

$ pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu121

RUN THE DIARISATION WORKFLOW

10. Place your mp3 files in a folder called samples and run the main script

$ python diarize_whisper_stablets_hdbscan.py.py


OPTIONAL

If you need WAV samples that do not need to be converted in order 
to test NeMo prior to testing the diarisation workflow:

https://huggingface.co/nvidia/speakerverification_en_titanet_large

$ sudo pacman -S git-lfs
$ git lfs install
$ git clone https://huggingface.co/datasets/espnet/an4
$ cd an4
$ tar xvf an4_sphere.tar.gz

The samples used in the example are (sph is essentially wav):

~/Downloads/an4/an4/wav/an4_clstk/fash/an255-fash-b.sph
~/Downloads/an4/an4/wav/an4_clstk/fash/cen7-fash-b.sph







