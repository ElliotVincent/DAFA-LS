<div align="center">
<h2>
Detecting Looted Archaeological Sites from Satellite Image Time Series

<a href="https://imagine.enpc.fr/~elliot.vincent/">Elliot Vincent</a>&emsp;
<a href="https://iconem.fr/">Merhaïl Saroufim</a>&emsp;
<a href="https://iconem.fr/">Jonathan Chemla</a>&emsp;  
<a href="https://iconem.fr/">Yves Ubelmann</a>&emsp;
<a href="https://www.e-patrimoines.org/patrimoine/new_interactive_ress/la-delegation-archeologique-francaise-en-afghanistan/">Philippe Marquis</a>&emsp;
<a href="https://www.di.ens.fr/~ponce/">Jean Ponce</a>&emsp;
<a href="https://imagine.enpc.fr/~aubrym/">Mathieu Aubry</a>

<p></p>

</h2>
</div>

Official PyTorch implementation of [**Detecting Looted Archaeological Sites from Satellite Image Time Series**](http://arxiv.org/abs/2409.09432).
Check out our [**webpage**](https://imagine.enpc.fr/~elliot.vincent/dafals) for other details!

We introduce the DAFA Looted Sites dataset (DAFA-LS), a labeled multi-temporal remote sensing dataset containing 55,480 images acquired monthly over 8 years across 675 Afghan archaeological sites, including 135 sites looted during the acquisition period. DAFA-LS is an interesting playground to assess the performance of satellite image time series (SITS) classification methods on a real and important use case.

![alt text](https://github.com/ElliotVincent/DAFA-LS/blob/main/dafals_teaser.png?raw=true)

If you find this code useful, don't forget to <b>star the repo :star:</b>.


## Installation :gear:

### 1. Clone the repository in recursive mode

```
git clone git@github.com:ElliotVincent/DAFA-LS.git --recursive
```

### 2. Download the datasets

You can download the datasets using the code below or by following [this link](https://drive.google.com/file/d/16v7_AcRwNeRhCacmQuX2477VYs51f4fU/view) (426M).

```
cd DAFA-LS
mkdir datasets
cd datasets
gdown 16v7_AcRwNeRhCacmQuX2477VYs51f4fU
unzip DAFA_LS.zip
```

### 3. Download pretrained weights for DOFA [1], SatMAE [10] and Scale-MAE [11]

```
cd ..
mkdir weights
cd weights
wget https://huggingface.co/XShadow/DOFA/resolve/main/DOFA_ViT_base_e100.pth?download=true
wget https://zenodo.org/record/7369797/files/fmow_pretrain.pth
wget https://github.com/bair-climate-initiative/scale-mae/releases/download/base-800/scalemae-vitlarge-800.pth
```

### 4. Create and activate virtual environment

```
python3 -m venv dafals
source dafals/bin/activate
python3 -m pip install -r requirements.txt
```
This implementation uses Pytorch.

## How to use 🚀

If you use the repository for the first time, please create a `results` folder:
```
mkdir results
```
Now you can run the following command, replacing `<config_name>` by either `resnet` [2], `dofa` [1], `satmae` [10], `scalemae` [11], `ltae` [3], `tempcnn` [4], `duplo` [5], `transformer` [6], `utae` [7], `tsvit_cls` [8], `tsvit_seg` [8], `pse_ltae` [9], `dofa_ltae` [1,3], `satmae_ltae` [10,3] or `scalemae_ltae` [11,3].
Replace `<exp_name>` by the experiment name of your choice. Output files will be located at `results/<exp_name>/`.
```
PYTHONPATH=$PYTHONPATH:./src python src/trainer.py -t <exp_name> -c <config_name>.yaml
```

## Citing

If you use our work in your project please cite:

```bibtex
@article{vincent2024detecting,
    title = {Detecting Looted Archaeological Sites from Satellite Image Time Serie},
    author = {Vincent, Elliot and Saroufim, Mehraïl and Chemla, Jonathan and Ubelmann, Yves and Marquis, Philippe and Ponce, Jean and Aubry, Mathieu},
    journal = {arXiv},
    year = {2024},
  }
```

And if you use our dataset, please give proper attributions to Planet Labs:

```bibtex
@article{planet2024planet,
    author={{Planet Team}},
    title={{Planet Application Program Interface: In Space for Life on Earth (San Francisco, CA)}},
    journal={\url{https://api.planet.com}},
    year={2024}
}
```

## Bibliography

[1] Z. Xiong et al. _Neural plasticity-inspired foundation model for observing the Earth crossing modalities_. (2024)  
[2] K. He et al. _Deep residual learning for image recognition_. (2016)  
[3] V. S. F. Garnot et al. _Lightweight temporal self-attention for classifying satellite images time series_. (2020)  
[4] C. Pelletier et al. _Temporal convolutional neural network for the classification of satellite image time series_. (2019)  
[5] R. Interdonato et al. _Duplo: A dual view point deep learning architecture for time series classification_. (2019)  
[6] M. Rußwurm et al. _Self-attention for raw optical satellite time series classification_. (2020)  
[7] V. S. F. Garnot et al. _Panoptic segmentation of satellite image time series with convolutional temporal attention networks_. (2021)  
[8] M. Tarasiou et al. _Vits for sits: Vision transformers for satellite image time series_. (2023)  
[9] V. S. F. Garnot et al. _Satellite image time series classification with pixel-set encoders and temporal self-attention_. (2020)  
[10] Y. Cong et al. _Satmae: Pre-training transformers for temporal and multi-spectral satellite imagery_. (2022)  
[11] C. Reed et al. _Scale-mae: A scale-aware masked autoencoder for multiscale geospatial representation learning_. (2023)
