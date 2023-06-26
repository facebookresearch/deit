# Sparsity Research on ViT

This reposity contains the PyTorch training code for the original DeiT models. Currently the code base are forked from the official [DeiT repo](https://github.com/facebookresearch/deit)

Here, I have build an interface and add some naive methods for add sparsity into the ViT.

## Sparsity NAS Training scripts
- Normal command
    - training
        ```
        python -m torch.distributed.launch --master_port 29510 --nproc_per_node=2 --use_env main.py \
        --model Sparse_deit_small_patch16_224 \
        --data-path /dataset/imagenet \
        --epochs 150 \
        --batch-size 128 \
        --pretrained \
        --lr 5e-5 \ 
        --min-lr 1e-6 \
        --nas-mode \
        --nas-config configs/deit_small_nxm_uniform24.yaml \
        --nas-test-config 2 4 \
        --output_dir nas_uniform_24_150epoch \
        --wandb
        ```
    - eval 
        ```
        python -m torch.distributed.launch --master_port 29510 --nproc_per_node=2 --use_env main.py \
        --model Sparse_deit_small_patch16_224
        --data-path /dataset/imagenet \
        --nas-mode \
        --nas-config configs/deit_small_nxm_uniform24.yaml \
        --nas-weights nas_uniform_24_150epoch/best_checkpoint.pth \
        --nas-test-config 2 4 \
        --eval
        ```
- KD command
    - training
        ```
        python -m torch.distributed.launch --master_port 29510 --nproc_per_node=2 --use_env main.py \
        --model Sparse_deit_small_patch16_224 \
        --data-path /dataset/imagenet \
        --epochs 150 \
        --batch-size 128 \
        --pretrained \\
        --lr 5e-5 \ 
        --min-lr 1e-6 \
        --nas-mode \
        --nas-config configs/deit_small_nxm_nas_1234.yaml \
        --nas-test-config 2 4 \
        --output_dir KD_nas_124+13_150epoch \
        --teacher-model deit_small_patch16_224 \
        --distillation-type soft \
        --distillation-alpha 1.0 \
        --wandb
        ```
    - eval 
        ```
        python -m torch.distributed.launch --master_port 29510 --nproc_per_node=2 --use_env main.py \
        --model Sparse_deit_small_patch16_224
        --data-path /dataset/imagenet \
        --nas-mode \
        --nas-config configs/deit_small_nxm_uniform24.yaml \
        --nas-weights KD_nas_124+13_150epoch/checkpoint.pth \
        --nas-test-config 2 4 \
        --eval
        ```
- Cifar-100 command
    - training
        ```
        python main.py \
            --model deit_small_patch16_224 \
            --batch-size 256 \
            --finetune https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth \
            --data-set CIFAR \
            --data-path /dataset/cifar100 \
            --opt sgd \
            --weight-decay 1e-4 \
            --lr 1e-2 \
            --output_dir deit_s_224_cifar_100 \
            --epochs 500
        ```

## Support Sparsity Searching Algorithm
Currently, we support the following sparsity strategy:
+ `lamp` : pruning via lamp score [paper](https://arxiv.org/abs/2010.07611)
+ `glob` : global pruning 
+ `unif` : uniform pruning
+ `unifplus` : uniform pruning with some specific modificaiton (i.e. no pruning the first and last layer)
+ `erk` : Erdos-Renyi-Kernel [paper](https://arxiv.org/pdf/1911.11134.pdf)

All of the support sparsity algorithm can be found in `./sparsity_factory/pruners.py`. 

The abovementioned methods will calculate the layer wise sparsity automatically once given the global target sparsity. In the following section, we will demonstrate how to use a custom designed sparsity level to sparsify the model

## Use custom layer-wise Sparsity

We can provide a custom config that define the target sparsity of each layer. 
Currently, we support two kind of sparsity including `nxm` and `unstructuted`.
User can create a `yaml` file the descibe the detail and pass into the main function by add the `--custom-config [path to config file]` argument when you call the `main.py`

## Example Usage (Pruning method)
To run a DeiT-S with custom configuration and eval the accuracy before finetuning
```
python main.py \ 
--model deit_small_patch16_224 \
--data-path [Path to imagenet] \
--output_dir [Path to output directory] \
--eval  \
--pruner custom \
--custom-config configs/deit_small_nxm.yaml
```

To finetune the DeiT-S with custom configuration
```
python main.py \ 
--model deit_small_patch16_224 \
--data-path [Path to imagenet] \
--output_dir [Path to output directory] \
--pruner custom \
--custom-config configs/deit_small_nxm.yaml
```


To use the algorithm to calculate the layer-sparsity and finetune given the global target sparsity to be 65%
```
python main.py \ 
--model deit_small_patch16_224 \
--data-path [Path to imagenet] \
--output_dir [Path to output directory] \
--pruner lamp \
--sparsity 0.65
```






