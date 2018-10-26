# FashionAI2018-TianChi

## Introduction
This is the main **Gluon** code of [阿里天池竞赛——服饰属性标签识别](https://tianchi.aliyun.com/competition/information.htm?spm=5176.100067.5678.2.505c3a26Oet3cf&raceId=231649). Note that this code is just a part of our final code, but provides the one of our **best single model**. Final submission is a ensemble model of two model: One is this resnet152v2 model ,and the other is the Inceptionv4 model from my teammates.
The code is based on [hetong007's code](https://github.com/hetong007/Gluon-FashionAI-Attributes) which provides a good baseline in the competition. This is my first time to use Gluon and thanks to hetong007~

Rank: 10/2950 (Season1)    17/2950 (Season2)

## Software:
- ubuntu14.04，cuda8.0，cudnn6.5
- python2.7
- mxnet-cu80
- numpy
- pandas


## Highlights
- **Higher performance:** Improve the result by many modifications for a **pure single model**(without backbone ensemble).
- **Faster training speed:** Use `gluon.data.vision.transforms` for data augmentation which is faster than original code.
- **Soft label:** Define our own cutom dataset and treat `'maybe'(m) label` as `'soft label'` when training which can boost the result.
- **Muti-scale train & test:** Use more scale augmentations when training and testing, especially for TTA(Test time augmentation). 
- **mAP defination:** Define mAP by ourselves according the competetion illustrate.
- **Random erasing:** Gluon version code defined by ourselvers.


## Training in a few lines

1. Download [data](https://tianchi.aliyun.com/competition/information.htm?spm=5176.100067.5678.2.505c3a26Oet3cf&raceId=231649) and extract it into `data1/`(season1) and `data2/`(season2).
2. `python2 prepare_data.py` Prepare dataset for trainset, valset and testset.
3. `bash benchmark.sh`
  - `num_gpus`，set to 1 for single GPU training
