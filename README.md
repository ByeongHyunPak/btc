# B-spline Texture Coefficients Estimator for Screen Content Image Super-Resolution
This repoository is the official pytorch implementation of **BTC** introduced by:
### [**B-spline Texture Coefficients Estimator for Screen Content Image Super-Resolution (CVPR 2023, Highlight)**](https://openaccess.thecvf.com/content/CVPR2023/papers/Pak_B-Spline_Texture_Coefficients_Estimator_for_Screen_Content_Image_Super-Resolution_CVPR_2023_paper.pdf)

## Quick start

1. Download a SCI1K pre-trained model:
[**RDN-BTC**](https://www.dropbox.com/s/fc6lzwd46ogszcw/rdn%2Bbtc-3rd.pth?dl=0)

## Environment
* Python 3
* Pytorch 1.13.0
* TensorboardX
* pyyaml, numpy, tqdm, imageio

## Dataset

1. `mkdir ../Data` for putting the dataset folders.

2. `cd ../Data` and download the datasets (**SCI1K**, **SCID**, and **SIQAD**) from [this repo](https://github.com/codyshen0000/ITSRN/tree/main/Data).

3. For the additional benchmarks in Tab 6, follow **Data** instruction provided by [this repo](https://github.com/yinboc/liif).


## Train & Test
* Train : `python train.py --config configs/train/[TRAIN_CONFIG] --gpu [GPU]`
  * `[TRAIN_CONFIG]` : to define model configuration (e.g. `train-rdn+btc-3rd.yaml`).
  * `[GPU]` : to specify the GPUS (e.g. `--gpu 0` or `--gpu 0,1`).
  
* Test : `python test.py --config configs/test/[TEST_CONFIG] --model save/[MODEL] --gpu [GPU]`
  * `[TEST_CONFIG]` : to define test configuration (e.g. `test-sci1k-02.yaml` for SCI1K dataset on x2).
  * `[MODEL]` : to define the pre-trained model (e.g. `rdn+btc-3rd/epoch_last.pth`).
  * `[GPU]` : to specify the GPUS (e.g. `--gpu 0` or `--gpu 0,1`).

## Acknowledgements
This code is built on [LIIF](https://github.com/yinboc/liif) and [LTE](https://github.com/jaewon-lee-b/lte).
We thank the authors for sharing their codes.
