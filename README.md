# DeiT: Data-efficient Image Transformers

This repository contains PyTorch evaluation code, training code and pretrained models for DeiT (Data-Efficient Image Transformers).

They obtain competitive tradeoffs in terms of speed / precision:

![DeiT](.github/deit.png)

For details see [Training data-efficient image transformers & distillation through attention](https://arxiv.org/abs/2012.12877) by Hugo Touvron, Matthieu Cord, Matthijs Douze, Francisco Massa, Alexandre Sablayrolles and Hervé Jégou. 

If you use this code for a paper please cite:

```
@article{touvron2020deit,
  title={Training data-efficient image transformers & distillation through attention},
  author={Hugo Touvron and Matthieu Cord and Matthijs Douze and Francisco Massa and Alexandre Sablayrolles and Herv\'e J\'egou},
  journal={arXiv preprint arXiv:2012.12877},
  year={2020}
}
```

# Model Zoo

We provide baseline DeiT models pretrained on ImageNet 2012.

| name | acc@1 | acc@5 | url |
| --- | --- | --- | --- |
| DeiT-tiny | 72.2 | 91.1 | [model](https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth) |
| DeiT-small | 79.9 | 95.0 | [model](https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth) |
| DeiT-base | 81.8 | 95.6 | [model](https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth) |


The models are also available via torch hub.
Before using it, make sure you have [`timm==0.3.2`](https://github.com/rwightman/pytorch-image-models) installed. To load DeiT-base with pretrained weights on ImageNet simply do:

```python
import torch
# check you have the right version of timm
import timm
assert timm.__version__ == "0.3.2"

# now load it with torchhub
model = torch.hub.load('facebookresearch/deit:main', 'deit_base_patch16_224', pretrained=True)
```

# Usage

First, clone the repository locally:
```
git clone https://github.com/facebookresearch/deit.git
```
Then, install PyTorch 1.7.0+ and torchvision 0.8.1+ and [pytorch-image-models 0.3.2](https://github.com/rwightman/pytorch-image-models):

```
conda install -c pytorch pytorch torchvision
pip install timm==0.3.2
```

## Data preparation

Download and extract ImageNet train and val images from http://image-net.org/.
The directory structure is the standard layout for the torchvision [`datasets.ImageFolder`](https://pytorch.org/docs/stable/torchvision/datasets.html#imagefolder), and the training and validation data is expected to be in the `train/` folder and `val` folder respectively:

```
/path/to/imagenet/
  train/
    class1/
      img1.jpeg
    class2/
      img2.jpeg
  val/
    class1/
      img3.jpeg
    class/2
      img4.jpeg
```

## Evaluation
To evaluate a pre-trained DeiT-base on ImageNet val with a single GPU run:
```
python main.py --eval --resume https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth --data-path /path/to/imagenet
```
This should give
```
* Acc@1 81.846 Acc@5 95.594 loss 0.820
```

For Deit-small, run:
```
python main.py --eval --resume https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth --model deit_small_patch16_224 --data-path /path/to/imagenet
```
giving
```
* Acc@1 79.854 Acc@5 94.968 loss 0.881
```

And for Deit-tiny:
```
python main.py --eval --resume https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth --model deit_tiny_patch16_224 --data-path /path/to/imagenet
```
which should give
```
* Acc@1 72.202 Acc@5 91.124 loss 1.219
```

## Training
To train DeiT-small and Deit-tiny on ImageNet on a single node with 4 gpus for 300 epochs run:

DeiT-small
```
python -m torch.distributed.launch --nproc_per_node=4 --use_env main.py --model deit_small_patch16_224 --batch-size 256 --data-path /path/to/imagenet
```

DeiT-tiny
```
python -m torch.distributed.launch --nproc_per_node=4 --use_env main.py --model deit_tiny_patch16_224 --batch-size 256 --data-path /path/to/imagenet
```

### Multinode training

Distributed training is available via Slurm and [submitit](https://github.com/facebookincubator/submitit):

```
pip install submitit
```

To train DeiT-base model on ImageNet on 2 nodes with 8 gpus each for 300 epochs:

```
python run_with_submitit.py --model deit_base_patch16_224 --data-path /path/to/imagenet
```

# License
This repository is released under the CC-BY-NC 4.0. license as found in the [LICENSE](LICENSE) file.

# Contributing
We actively welcome your pull requests! Please see [CONTRIBUTING.md](.github/CONTRIBUTING.md) and [CODE_OF_CONDUCT.md](.github/CODE_OF_CONDUCT.md) for more info.
