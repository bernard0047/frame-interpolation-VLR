<img width="929" alt="image" src="https://github.com/bernard0047/frame-interpolation-VLR/assets/81643693/ec168518-e338-4c28-bdec-f18bd77170ad">



download ours_t.pkl from https://drive.google.com/drive/folders/16jUa3HkQ85Z5lb5gce1yoaWkP-rdCd0o
paste it at frame-interpolation/infer_models/emavfi/ours_t.pkl


# Extracting Motion and Appearance via Inter-Frame Attention for Efficient Video Frame Interpolation [arxiv](https://arxiv.org/abs/2303.00440)

> [**Extracting Motion and Appearance via Inter-Frame Attention for Efficient Video Frame Interpolation**](https://arxiv.org/abs/2303.00440)<br>
> Accepted by **CVPR 2023**<br>
> [Guozhen Zhang](https://github.com/GuozhenZhang1999), [Yuhan Zhu](https://github.com/Breeze-zyuhan), [Haonan Wang](https://github.com/haonanwang0522), Youxin Chen, [Gangshan Wu](http://mcg.nju.edu.cn/member/gswu/en/index.html), [Limin Wang](http://wanglimin.github.io/)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/extracting-motion-and-appearance-via-inter/video-frame-interpolation-on-vimeo90k)](https://paperswithcode.com/sota/video-frame-interpolation-on-vimeo90k?p=extracting-motion-and-appearance-via-inter)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/extracting-motion-and-appearance-via-inter/video-frame-interpolation-on-ucf101-1)](https://paperswithcode.com/sota/video-frame-interpolation-on-ucf101-1?p=extracting-motion-and-appearance-via-inter)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/extracting-motion-and-appearance-via-inter/video-frame-interpolation-on-xiph-4k-1)](https://paperswithcode.com/sota/video-frame-interpolation-on-xiph-4k-1?p=extracting-motion-and-appearance-via-inter)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/extracting-motion-and-appearance-via-inter/video-frame-interpolation-on-xiph-2k)](https://paperswithcode.com/sota/video-frame-interpolation-on-xiph-2k?p=extracting-motion-and-appearance-via-inter)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/extracting-motion-and-appearance-via-inter/video-frame-interpolation-on-x4k1000fps-2k)](https://paperswithcode.com/sota/video-frame-interpolation-on-x4k1000fps-2k?p=extracting-motion-and-appearance-via-inter)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/extracting-motion-and-appearance-via-inter/video-frame-interpolation-on-x4k1000fps)](https://paperswithcode.com/sota/video-frame-interpolation-on-x4k1000fps?p=extracting-motion-and-appearance-via-inter)

<div align="center">
  <img src="figs/EMA-VFI.png" width="1000"/>
</div>



## :satisfied: HighLights

In this work, we propose to exploit inter-frame attention for extracting motion and appearance information in video frame interpolation. In particular, we utilize the correlation information hidden within the attention map to simultaneously enhance the appearance information and model motion. Meanwhile, we devise an hybrid CNN and Transformer framework to achieve a better trade-off between performance and efficiency. Experiment results show that our proposed module achieves state-of-the-art performance on both fixed- and arbitrary-timestep interpolation and enjoys effectiveness compared with the previous SOTA method.

Runtime and memory usage compared with previous SOTA method:
<div align="center">
  <img src=figs/time.png width=400 />
</div>

## :two_hearts:Dependencies

- torch 1.8.0
- python 3.8
- skimage 0.19.2
- numpy 1.23.1
- opencv-python 4.6.0
- timm 0.6.11
- tqdm

## :sunglasses:	Play with Demos

1. Download the [model checkpoints](https://drive.google.com/drive/folders/16jUa3HkQ85Z5lb5gce1yoaWkP-rdCd0o?usp=sharing) ([baidu](https://pan.baidu.com/s/1kvxubOrCxq2Mjc6SXZSa0w)&code:gi5j)and put the ```ckpt``` folder into the root dir.
2. Run the following commands to generate 2x and Nx (arbitrary) frame interpolation demos:

```shell
python demo_2x.py        # for 2x interpolation
python demo_Nx.py --n 8  # for 8x interpolation
```

By running above commands, you should get the follow examples by default:

<p float="left">
  <img src=figs/out_2x.gif width=340 />
  <img src=figs/out_Nx.gif width=340 /> 
</p>

## :sparkles:	Training for Fixed-timestep Interpolation

1. Download [Vimeo90K dataset](http://toflow.csail.mit.edu/)
2. Run the following command at the root dir:

```shell
  python -m torch.distributed.launch --nproc_per_node=4 train.py --world_size 4 --batch_size 8 --data_path **YOUR_VIMEO_DATASET_PATH** 
```

The default training setting is *Ours*. If you want train *Ours_small* or your own model, you can modify the ```MODEL_CONFIG``` in  ```config.py```.

## :runner:	Evaluation

1. Download the co3d Dataset Subset:

Dataset

We use a subset of the Common Objects in 3D (Co3D) dataset having:
88 objects from 48 distinct categories with 202 images each .


2. Download the [model checkpoints]() and put the ```ckpt``` folder into the root dir.



## :muscle:	Citation



Reference to below paper

```
@inproceedings{zhang2023extracting,
  title={Extracting motion and appearance via inter-frame attention for efficient video frame interpolation},
  author={Zhang, Guozhen and Zhu, Yuhan and Wang, Haonan and Chen, Youxin and Wu, Gangshan and Wang, Limin},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={5682--5692},
  year={2023}
}
```

## :heartpulse:	License and Acknowledgement

This project is released under the Apache 2.0 license. The codes are based on [RIFE](https://github.com/hzwer/arXiv2020-RIFE), [PvT](https://github.com/whai362/PVT), [IFRNet](https://github.com/ltkong218/IFRNet), [Swin](https://github.com/microsoft/Swin-Transformer) and [HRFormer](https://github.com/HRNet/HRFormer). Please also follow their licenses. Thanks for their awesome works.
