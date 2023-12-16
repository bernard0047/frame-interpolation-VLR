<img width="929" alt="image" src="https://github.com/bernard0047/frame-interpolation-VLR/assets/81643693/ec168518-e338-4c28-bdec-f18bd77170ad">



## :satisfied: HighLights

Generating 3D models from sparse views presents a formidable challenge in computer vision. Recent advancements, including various iterations of Neural Radiance Fields (NeRF), have struggled to achieve high fidelity using limited input images. While some contemporary methods have employed diffusion-based techniques to augment sparse views, their effectiveness varies. These methods often perform fairly on specific datasets but typically lack a robust semantic foundation for broader real-world applicability. Additionally, they tend to produce images with compromised fidelity, resulting in blurred reconstructions when viewed from different angles.

This work proposes an alternative strategy to address these limitations by interpolating existing views. Our approach uses these processed views as additional data points for NeRF generation. By exploring this methodology, we aim to enhance the fidelity and clarity of 3D reconstructions derived from limited viewpoints. This research not only contributes to overcoming the challenges posed by sparse view inputs but also extends the practical applicability of NeRF in more diverse real-world scenarios. We build an end-to-end pipeline to create faithful reconstructions from a sparse set of views of a given subject.

## :two_hearts:Dependencies

- torch 1.8.0
- python 3.8
- skimage 0.19.2
- numpy 1.23.1
- opencv-python 4.6.0
- timm 0.6.11
- tqdm

## :sunglasses:	Play with Demos

1. Run the following commands to generate 2x and Nx (arbitrary) frame interpolation demos:

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

1. Download [Co3D dataset](https://github.com/facebookresearch/co3d)
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

## :heartpulse:	License and Acknowledgement

This project is released under the Apache 2.0 license. The codes are based on [RIFE](https://github.com/hzwer/arXiv2020-RIFE), [PvT](https://github.com/whai362/PVT), [IFRNet](https://github.com/ltkong218/IFRNet), [Swin](https://github.com/microsoft/Swin-Transformer) and [HRFormer](https://github.com/HRNet/HRFormer). Please also follow their licenses. Thanks for their awesome works.
