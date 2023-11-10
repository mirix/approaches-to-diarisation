CREATE A VIRTUAL ENVIRONMENT

For instance, on Arch Linux or derivatives:

1. Install yay (optional):

```
sudo pacman -S yay
```

2. Install Python 3.10:

```
yay python310
```

3. Create a virtual environment with Python 3.10 (I called it nemo10):

```
cd ~/Downloads
python3.8 -m venv nemo10
```

4. Activate the new environment:

```
source ~/Downloads/nemo10/bin/activate
```

For verification purposes:

```
python --version
```

In my case it outputs Python 3.11.5 outside of the environment and Python 3.10.13 inside of the environment (what matters is just the major and the minor, 3.10, the patch, 13, is irrelevant)

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

NOTE: If you are using the RAPIDS version you can delete the lines reading hdbscan and umap-learn. These are only required for the, now deprecated, CPU version.

7. Install demucs

```
sudo pacman -S sox libid3tag libmad twolame
(for debian, sudo apt-get install sox libsox-fmt-mp3)

python -m pip install -U "git+https://github.com/facebookresearch/demucs#egg=demucs"
```

8. Install RAPIDS (only the required modules)

```
pip install --extra-index-url=https://pypi.nvidia.com cudf-cu12 cuml-cu12
```

NOTE: This is only required to run the RAPIDS script, omit this step if you are running the CPU-only script.

9. Install Nvidia NeMo

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

NOTE: Actually we only need two modules: asr and nlp.

(OPTIONAL, ONLY IF THE PREVIOUS COMMAND FAILS)

The above fails from a venv at the time of this writing. If it still does for you, try the following hack:

```
git clone https://github.com/NVIDIA/NeMo
cd NeMo
```
Edit requirement/requirements_nlp.txt and replace "fasttext" with "fasttext-wheel" (no quotation marks), then:

```
pip install -e ".[all]"
```

10. Reinstall Pytorch:

gpu (stable with CUDA 11.8):

```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
NOTE: In a CPU-only environment:

```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

RUN THE DIARISATION WORKFLOW

11. Place your mp3 files in a folder called samples and run the main script

```
python diarize_whisper_stablets_nemo_hdbscan_rapids.py
```


TROUBLESHOOTING

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






