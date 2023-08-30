CREATE A VIRTUAL ENVIRONMENT

For instance, on Arch Linux or derivatives:

1. Install yay (optional):

```
sudo pacman -S yay
```

2. Install Python 3.8:

```
yay python38
```

3. Create a virtual environment with Python 3.8 (I called it nemo):

```
cd ~/Downloads
python3.8 -m venv nemo
```

4. Activate the new environment:

```
source ~/Downloads/nemo/bin/activate
```

For verification purposes:

```
python --version
```

In my case it outputs Python 3.11.3 outside of the environment and Python 3.8.18 inside of the environment (what matters is just the major and the minor, 3.8, the patch, 18, is irrelevant)

CLONE THE REPO AND INSTALL THE REQUIREMENTS

5. Clone this repo:

```
git clone https://github.com/mirix/approaches-to-diarisation.git
cd approaches-to-diarisation
```

6. Install the requirements

```
pip install -r requirements.txt
```

7. Install demucs

```
sudo pacman -S sox libid3tag libmad twolame
(for debian, sudo apt-get install sox libsox-fmt-mp3)

python -m pip install -U "git+https://github.com/facebookresearch/demucs#egg=demucs"
```

8. Install Nvidia NeMo

```
sudo pacman -S libsndfile ffmpeg
```
(for debian, sudo apt-get install libsndfile1 ffmpeg)

```
pip install cython
```
First, let's try the pip way:

```
python -m pip install "git+https://github.com/NVIDIA/NeMo.git@main#egg=nemo_toolkit[all]"
```

Actually we only need two modules: asr and nlp.

(OPTIONAL, ONLY IF THE PREVIOUS COMMAND FAILS)

The above fails at the time of this writing. If it still does for you, try the following hack:

```
git clone https://github.com/NVIDIA/NeMo
cd NeMo
```
Edit requirement/requirements_nlp.txt and replace "fasttext" with "fasttext-wheel" (no quotation marks), then:

```
pip install -e ".[all]"
```

9. Reinstall Pytorch (OPTIONAL, only if you want a version that is different from the one pulled by NeMo):

cpu (nightly):

```
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cpu
```

gpu (nightly for CUDA 12.1):

```
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu121
```

RUN THE DIARISATION WORKFLOW

10. Place your mp3 files in a folder called samples and run the main script

```
python diarize_whisper_stablets_nemo_hdbscan.py
```


OPTIONAL

If you run into problems with NeMo, you may wish to test your installation prior to running the diarisation workflow.

You can use the following code:

https://huggingface.co/nvidia/speakerverification_en_titanet_large

The wav samples can be obtained as follows:

```
sudo pacman -S git-lfs
git lfs install
cd ~/Downloads
git clone https://huggingface.co/datasets/espnet/an4
cd an4
tar xvf an4_sphere.tar.gz
```

The exact samples used in the Huggingface example are the following (sph is essentially wav):

```
~/Downloads/an4/an4/wav/an4_clstk/fash/an255-fash-b.sph
~/Downloads/an4/an4/wav/an4_clstk/fash/cen7-fash-b.sph
```






