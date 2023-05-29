# Dynamic Local Aggregation Network with Adaptive Clusterer for Anomaly Detection (ECCV 2022)

This repository is the official implementation of the ECCV 2022 paper "Dynamic Local Aggregation Network with Adaptive Clusterer for Anomaly Detection" with pytorch. [[PDF](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136640398.pdf)] [[Arxiv](https://arxiv.org/abs/2207.10948)]

## Requirements

* Python 3.8
* PyTorch 1.9.0
* Numpy
* Sklearn
* minisom (https://github.com/JustGlowing/minisom)
## Datasets
* USCD Ped2 [[dataset](https://github.com/StevenLiuWen/ano_pred_cvpr2018)]
* CUHK Avenue [[dataset](https://github.com/StevenLiuWen/ano_pred_cvpr2018)]
* ShanghaiTech [[dataset](https://github.com/StevenLiuWen/ano_pred_cvpr2018)]

These datasets are from an official github of "Future Frame Prediction for Anomaly Detection - A New Baseline (CVPR 2018)".

Download the datasets into ``data`` folder, like ``./data/ped2/``

## AC pre-clustering
```bash
git clone https://github.com/Beyond-Zw/DLAN-AC.git
cd projects/DLAN-AC
python train.py --AC_clustering True --dataset_path 'your_dataset_directory' --dataset_type ped2 --exp_dir 'your_log_directory'
```
## Formal training
```bash
python train.py --AC_clustering False --dataset_path 'your_dataset_directory' --dataset_type ped2 --exp_dir 'your_log_directory'
```
## Citation

If you feel this work helpful, please cite our paper:

```
@article{yang2023video,
  title={Video Event Restoration Based on Keyframes for Video Anomaly Detection},
  author={Yang, Zhiwei and Liu, Jing and Wu, Zhaoyang and Wu, Peng and Liu, Xiaotao},
  journal={arXiv preprint arXiv:2304.05112},
  year={2023}
}
```

## Acknowledgement
This repository contains modified codes from:

* [[MNAD](https://github.com/cvlab-yonsei/MNAD)]
* [[MPN](https://github.com/ktr-hubrt/MPN)]

We sincerely thank the owners of all these great repos!
