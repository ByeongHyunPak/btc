# B-spline Texture Coefficients Estimator for Screen Content Image Super-Resolution
This repoository is the official pytorch implementation of BTC introduced by:

**B-spline Texture Coefficients Estimator for Screen Content Image Super-Resolution (CVPR 2023)**

## Environment
Our code is based on python 3.6, ubuntu 20.04, pytorch 1.10.0, and CUDA 11.3 (NVIDIA RTX 3090 24GB, sm86).

For environmet setup,
```
conda env create --file environment.yaml
conda activate btc
```

## Dataset

1. `mkdir ../Data` for putting the dataset folders.

2. `cd ../Data` and download the datasets (**SCI1K**, **SCID**, and **SIQAD**) from [this repo](https://github.com/codyshen0000/ITSRN/tree/main/Data).

3. For the additional benchmarks in Tab 6, follow **Data** instruction provided by [this repo](https://github.com/yinboc/liif).


## Pre-trained model


## Train & Test
