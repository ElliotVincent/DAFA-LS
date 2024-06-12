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

Official PyTorch implementation of [**Detecting Looted Archaeological Sites from Satellite Image Time Series**](https://github.com/ElliotVincent/DAFA-LS).
Check out our [**webpage**](https://imagine.enpc.fr/~elliot.vincent/) for other details!

We introduce the DAFA Looted Sites dataset (DAFA-LS), a labeled multi-temporal remote sensing dataset containing 55,480 images acquired monthly over 8 years across 675 Afghan archaeological sites, including 135 sites looted during the acquisition period. DAFA-LS is an interesting playground to assess the performance of satellite image time series (SITS) classification methods on a real and important use case.

![alt text](https://github.com/ElliotVincent/DAFA-LS/blob/main/dafals_teaser.png?raw=true)

If you find this code useful, don't forget to <b>star the repo :star:</b>.


## Installation :gear:

### 1. Clone the repository in recursive mode

```
git clone git@github.com:ElliotVincent/DAFA-LS.git --recursive
```

### 2. Download the datasets

You can download the datasets using the code below or by following this link](https://drive.google.com/file/d/16v7_AcRwNeRhCacmQuX2477VYs51f4fU/view) (426M).

```
cd DAFA-LS
mkdir datasets
cd datasets
gdown 16v7_AcRwNeRhCacmQuX2477VYs51f4fU
unzip DAFA_LS.zip
```

### 3. Create and activate virtual environment

```
python3 -m venv dafals
source dafals/bin/activate
python3 -m pip install -r requirements.txt
```
This implementation uses Pytorch.

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
