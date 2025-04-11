# Paper Title: 
Adaptive Additive Parameter Updates of Vision Transformers for Few-Shot Continual Learning 


# Abstract:
Integrating new class information without losing previously acquired knowledge remains a central challenge in artificial intelligence, often referred to as catastrophic forgetting. Few-shot class incremental learning (FSCIL) addresses this by first training a model on a robust dataset of base classes and then incrementally adapting it in successive sessions using only a few labeled examples per novel class. However, this approach is prone to overfitting on the limited new data, which can compromise overall performance and exacerbate forgetting. In this work, we propose a simple yet effective novel FSCIL framework that leverages a frozen Vision Transformer (ViT) backbone augmented with parameter-efficient additive updates. Our approach freezes the pre-trained ViT parameters and selectively injects trainable weights into the self-attention modules via an additive update mechanism. This design updates only a small subset of parameters to accommodate new classes without sacrificing the representations learned during the base session. By fine-tuning a limited number of parameters, our method preserves the generalizable features in the frozen ViT while reducing the risk of overfitting. Furthermore, as most parameters remain fixed, the model avoids overwriting previously learned knowledge when small novel data batches are introduced. Extensive experiments on benchmark datasets demonstrate that our approach yields state-of-the-art performance compared to baseline FSCIL methods.

![My Image](Main_Figure.png)

# Datasets
Please follow the instructions for downloading datasets in [CEC](https://github.com/icoz69/CEC-CVPR2021?tab=readme-ov-file#datasets-and-pretrained-models).


# Running the Scripts

## Cifar-100
```bash  
python train.py -project main -dataset cifar100 -lr_base 0.1 -epochs_base 10 -gpu 0 --main --save main -batch_size_base 128 -seed 1  --temp 32
```
## CUB-200
```bash  
python train.py -project main -dataset cub200 -lr_base 0.01 -epochs_base 10 -gpu 0 --main --save main -batch_size_base 128 -seed 1 --temp 8
``` 
## miniImageNet
```bash  
python train.py -project main -dataset mini_imagenet -lr_base 0.01 -epochs_base 10 -gpu 0 --main --save main -batch_size_base 128 -seed 1 --temp 32
``` 

# Acknowledgement

We would like to thank the below repositories for their contributions. 

[CLOSER](https://github.com/JungHunOh/CLOSER_ECCV2024/tree/master?tab=readme-ov-file) 

[CEC](https://github.com/icoz69/CEC-CVPR2021?tab=readme-ov-file#datasets-and-pretrained-models)

[fscil](https://github.com/xyutao/fscil)
