# [CVPR 2023] B-spline Texture Coefficients Estimator for Screen Content Image Super-Resolution (Highlight)
### [**B-spline Texture Coefficients Estimator for Screen Content Image Super-Resolution**](https://openaccess.thecvf.com/content/CVPR2023/papers/Pak_B-Spline_Texture_Coefficients_Estimator_for_Screen_Content_Image_Super-Resolution_CVPR_2023_paper.pdf)
>[Byeonghyun Pak](https://byeonghyunpak.github.io)\*, [Jaewon Lee](https://scholar.google.com/citations?user=O91nuqIAAAAJ)\*, [Kyong Hwan Jin](https://scholar.google.com/citations?user=aLYNnyoAAAAJ)\
>Daegu Gyeongbuk Institute of Science and Technology (DGIST)\
>CVPR 2023, Highlight

#### [[`Project Page`](https://byeonghyunpak.github.io/btc/)] [[`Paper`](https://openaccess.thecvf.com/content/CVPR2023/papers/Pak_B-Spline_Texture_Coefficients_Estimator_for_Screen_Content_Image_Super-Resolution_CVPR_2023_paper.pdf)]


## Environment
### Requirements
* Python 3
* Pytorch 1.13.0
* TensorboardX
* pyyaml, numpy, tqdm, imageio

### Checkpoint
* Download a SCI1K pre-trained model:
[**RDN-BTC**](https://www.dropbox.com/scl/fi/et68vpowdoau4iqvg34n8/rdn-btc-3rd.pth?rlkey=vk7auq6dj0xdi7t652rahipwm&st=kec3iwly&dl=0)

## Datasets

1. `mkdir ../Data` for putting the dataset folders.

2. `cd ../Data` and download the datasets (**SCI1K**, **SCID**, and **SIQAD**) from [this repo](https://github.com/codyshen0000/ITSRN/tree/main/Data).

3. For the additional benchmarks in Tab 6, follow **Data** instruction provided by [this repo](https://github.com/yinboc/liif?tab=readme-ov-file#data).

## Demo
```
python demo.py --input [INPUT] --model [MODEL] --scale [SCALE] --output output.png --gpu [GPU]
```
 * `[INPUT]` : input image's path (e.g. `--input input.png`).
 * `[MODEL]` : to define the pre-trained model (e.g. `--model rdn+btc-3rd.pth`).
 * `[SCALE]` : arbitrary magnification (e.g. `--scale 3` or `--scale 6.4`).
 * `[GPU]` : to specify the GPUS (e.g. `--gpu 0`).

## Train
```
python train.py --config configs/train/[TRAIN_CONFIG] --gpu [GPU]
```
  * `[TRAIN_CONFIG]` : to define model configuration (e.g. `train-rdn+btc-3rd.yaml`).
  * `[GPU]` : to specify the GPUS (e.g. `--gpu 0` or `--gpu 0,1`).
## Train
```
python test.py --config configs/test/[TEST_CONFIG] --model save/[MODEL] --gpu [GPU]
```
  * `[TEST_CONFIG]` : to define test configuration (e.g. `test-sci1k-02.yaml` for SCI1K dataset on x2).
  * `[MODEL]` : to define the pre-trained model (e.g. `rdn+btc-3rd/epoch_last.pth`).
  * `[GPU]` : to specify the GPUS (e.g. `--gpu 0` or `--gpu 0,1`).

## Citation
If you find our code helpful, please cite our paper:
```bibtex
@inproceedings{pak2023b,
  title     = {Textual Query-Driven Mask Transformer for Domain Generalized Segmentation},
  author    = {Pak, Byeonghyun and Lee, Jaewon and Jin, Kyong Hwan},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages     = {10062--10071},
  year      = {2023}
}
```

## Acknowledgements
This project is based on the following open-source projects.
We thank the authors for sharing their codes.
- [LIIF](https://github.com/yinboc/liif)
- [LTE](https://github.com/jaewon-lee-b/lte)
