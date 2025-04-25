# network.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm  # make sure timm is installed: pip install timm
from torchvision import transforms
import numpy as np
import math


# This layer computes: output = (W0 + alpha * ΔW) x + bias,
# where W0 is frozen (pre-trained) and ΔW is trainable.
class AdditiveLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=False, alpha=1.0):
        super(AdditiveLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha

        # Initialize the original weight W0 and freeze it.
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.weight.requires_grad = False  # Freeze W0

        # Create a delta_weight parameter that will be replaced by shared one.
        self.delta_weight = nn.Parameter(torch.zeros(out_features, in_features))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, x):
        # Compute: output = (W0 + alpha * ΔW) x + bias
        return F.linear(x, self.weight + self.alpha * self.delta_weight, self.bias)


class MYNET(nn.Module):
    def __init__(self, args, mode=None):
        super(MYNET, self).__init__()
        self.mode = mode
        self.args = args

        # Create ViT backbone (vit_base_patch16_224) and remove its default classification head.
        #self.encoder = timm.create_model('vit_base_patch16_224', pretrained=True) #dagger
        self.encoder = timm.create_model('vit_base_patch16_224_in21k', pretrained=True) #double_dagger
        self.encoder.head = nn.Identity()

        # Freeze all encoder parameters.
        for param in self.encoder.parameters():
            param.requires_grad = False

        # Create a shared delta parameter for qkv layers.
        # Find one block that has a qkv layer to get the dimensions.
        for block in self.encoder.blocks:
            if hasattr(block.attn, 'qkv'):
                old_qkv = block.attn.qkv
                in_features = old_qkv.in_features
                out_features = old_qkv.out_features
                break
        self.shared_delta = nn.Parameter(torch.zeros(out_features, in_features))

        # Choose the blocks where we want to enable additive updates.
        num_blocks = len(self.encoder.blocks)
        num_delta_blocks = 12  # Number of transformer blocks to update
        for i, block in enumerate(self.encoder.blocks):
            if i >= num_blocks - num_delta_blocks:
                # Replace the attention module's qkv layer with our AdditiveLinear version.
                if hasattr(block.attn, 'qkv'):
                    old_qkv = block.attn.qkv
                    in_features = old_qkv.in_features
                    out_features = old_qkv.out_features
                    # Create a new Additive Update.
                    new_qkv = AdditiveLinear(in_features, out_features, bias=(old_qkv.bias is not None), alpha=1.0)
                    # Copy the original weights and biases so that initially ΔW = 0.
                    with torch.no_grad():
                        new_qkv.weight.copy_(old_qkv.weight)
                        if old_qkv.bias is not None:
                            new_qkv.bias.copy_(old_qkv.bias)
                    # Share the same delta parameter across all selected blocks.
                    new_qkv.delta_weight = self.shared_delta
                    # Replace the old qkv with the new one.
                    block.attn.qkv = new_qkv

        # Set the feature dimension from the ViT.
        self.feature_dim = self.encoder.embed_dim

        # Define the classifier head mapping the feature_dim to the number of classes.
        self.classifier = nn.Linear(self.feature_dim, self.args.num_classes, bias=False)

        self.session = 0
        self.test = False

    def forward_metric(self, x, pos=None):
        # Pass input x through the ViT encoder.
        x = self.encoder(x)  # shape: [B, feature_dim]
        fc = self.classifier.weight  # shape: [num_classes, feature_dim]
        # Compute cosine similarity between normalized features and classifier weights.
        x = F.linear(F.normalize(x, p=2, dim=-1), F.normalize(fc, p=2, dim=-1))
        x = self.args.temperature * x
        return x

    def encode(self, x):
        # Simply return the features from the ViT encoder.
        x = self.encoder(x)
        return x

    def forward(self, input, pos=None):
        if self.mode != 'encoder':
            return self.forward_metric(input, pos)
        elif self.mode == 'encoder':
            return self.encode(input)
        else:
            raise ValueError('Unknown mode')

    def update_fc(self, dataloader, class_list, session):
        """
        Updates the classifier for the classes in class_list by computing the average feature
        for each class from the data in the dataloader.
        """
        feats = []
        labels = []
        # Set up the appropriate transform based on the dataset.
        if self.args.dataset == 'cifar100':
            dataloader.dataset.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5071, 0.4865, 0.4409),
                                     std=(0.2673, 0.2564, 0.2762))
            ])
        elif self.args.dataset == 'mini_imagenet':
            dataloader.dataset.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        else:
            dataloader.dataset.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])

        with torch.no_grad():
            for batch in dataloader:
                data, label = [_.cuda() for _ in batch]
                feats.append(self.encoder(data).detach())
                labels.append(label)
        feats = torch.cat(feats, dim=0)
        labels = torch.cat(labels, dim=0)

        # For each unique label, update the classifier weight as the average feature vector.
        for ii in range(labels.unique().shape[0]):
            class_idx = labels.min() + ii
            self.classifier.weight.data[class_idx] = feats[labels == class_idx].mean(dim=0)
