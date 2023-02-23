# Sparsity Research on ViT

This reposity contains the PyTorch training code for the original DeiT models. Currently the code base are forked from the official [DeiT repo](https://github.com/facebookresearch/deit)

Here, I have build an interface and add some naive methods for add sparsity into the ViT.


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

## Example Usage
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






