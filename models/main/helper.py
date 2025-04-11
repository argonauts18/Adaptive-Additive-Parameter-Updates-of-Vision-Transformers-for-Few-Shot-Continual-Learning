# helper.py
import math
from .Network import MYNET
from utils import *
from tqdm import tqdm
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from torchvision import transforms

# ------------------------------
# Updated base_train function 
# ------------------------------
def base_train(model, trainloader, optimizer, scheduler, epoch, args):
    """
    Base training function using cross-entropy loss.
    This version extracts the [CLS] token from the ViT encoder output.
    
    Args:
        model: The model.
        trainloader: Data loader for training.
        optimizer, scheduler: Optimizer and scheduler.
        epoch: Current epoch.
        args: Command-line arguments.
    
    Returns:
        metrics: A list with averaged cross-entropy loss.
        ta: Overall training accuracy.
    """
    tl = Averager()  # overall loss accumulator
    ta = Averager()  # overall accuracy accumulator
    model = model.train()
    tqdm_gen = tqdm(trainloader)
    
    ce_total = 0

    for i, batch in enumerate(tqdm_gen, 1):
        data, train_label = [_ for _ in batch]
        # data is a list of augmented tensors, each of shape [B, C, H, W]
        B, C, H, W = data[0].shape
        # Concatenate all augmented views along the batch dimension and move to GPU.
        data = torch.cat(data, dim=0).cuda()
        train_label = train_label.cuda()
        
        # Use forward_features to get raw token embeddings.
        feats = model.module.encoder.forward_features(data)  # shape: [B*(1+num_aug), num_tokens, D]
        # Extract the [CLS] token (assumed to be the first token).
        cls_feats = feats[:, 0]  # shape: [B*(1+num_aug), D]
        cls_feats = F.normalize(cls_feats, dim=-1)
        
        # Compute logits using base classifier weights.
        base_weights = model.module.classifier.weight[:args.base_class].cuda()
        logits = F.linear(cls_feats, F.normalize(base_weights, p=2, dim=-1)) * args.temp
        
        # Compute cross-entropy loss.
        ce_loss = F.cross_entropy(logits, train_label.repeat(logits.shape[0] // B))
        loss = ce_loss
        
        ce_total += ce_loss.item()
        
        acc = count_acc(logits, train_label.repeat(logits.shape[0] // B))
        lrc = scheduler.get_last_lr()[0]
        tqdm_gen.set_description(
            'Session 0, epo {}, lrc={:.4f}, ce_loss={:.4f}, acc={:.4f}'.format(
                epoch, lrc, ce_loss.item(), acc))
        
        tl.add(loss.item())
        ta.add(acc)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    metrics = [ce_total / i]
    return metrics, ta.item()


# ------------------------------
# Updated replace_base_fc function 
# ------------------------------
def replace_base_fc(trainset, transform, model, args):
    """
    Replaces the classifier's weights with the average feature embeddings 
    computed on the base training data using a transform suited for ViT.
    Here, we extract the [CLS] token from the encoder output.
    """
    model = model.eval()
    trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=128,
                                              num_workers=8, pin_memory=True, shuffle=False)
    viT_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    trainloader.dataset.transform = viT_transform
    embedding_list = []
    label_list = []
    with torch.no_grad():
        for batch in trainloader:
            data, label = [_.cuda() for _ in batch]
            model.mode = 'encoder'
            # Use forward_features to get raw token embeddings.
            embedding = model.module.encoder.forward_features(data)  # shape: [B, num_tokens, D]
            cls_embedding = embedding[:, 0]  # extract [CLS] token: [B, D]
            embedding_list.append(cls_embedding.cpu())
            label_list.append(label.cpu())
    embedding_list = torch.cat(embedding_list, dim=0)
    label_list = torch.cat(label_list, dim=0)
    proto_list = []
    for class_index in range(args.base_class):
        data_index = (label_list == class_index).nonzero()
        embedding_this = embedding_list[data_index.squeeze(-1)]
        embedding_this = embedding_this.mean(0)
        proto_list.append(embedding_this)
    proto_list = torch.stack(proto_list, dim=0)
    model.module.classifier.weight.data[:args.base_class] = proto_list
    return model

# ------------------------------
# Updated test function 
# ------------------------------
def test(model, testloader, epoch, args, session):
    """
    Evaluates the model on the test set.
    In incremental sessions, separates performance on base and new classes.
    Assumes that the model's forward method computes logits using the [CLS] token.
    """
    test_class = args.base_class + session * args.way
    model = model.eval()
    vl = Averager()
    va_total = 0
    va_correct = 0
    if session > 0:
        va_base_total = 0
        va_base_correct = 0
        va_new_total = 0
        va_new_correct = 0
    model.session = session
    model.test = True
    with torch.no_grad():
        tqdm_gen = tqdm(testloader)
        for i, batch in enumerate(tqdm_gen, 1):
            data, test_label = [_.cuda() for _ in batch]
            logits = model(data)
            logits = logits[:, :test_class]
            loss = F.cross_entropy(logits, test_label)
            acc = count_acc(logits, test_label)
            vl.add(loss.item())
            va_total += logits.shape[0]
            va_correct += (logits.argmax(dim=-1) == test_label).sum()
            if session > 0:
                base_mask = test_label < args.base_class
                new_mask = test_label >= args.base_class
                va_base_total += base_mask.sum()
                va_new_total += new_mask.sum()
                va_base_correct += (logits.argmax(dim=-1)[base_mask] == test_label[base_mask]).sum()
                va_new_correct += (logits.argmax(dim=-1)[new_mask] == test_label[new_mask]).sum()
        vl = vl.item()
        va = va_correct / va_total
        if session > 0:
            va_base = va_base_correct / va_base_total
            va_new = va_new_correct / va_new_total
            print('epo {}, test, loss={:.4f} acc={:.4f}'.format(epoch, vl, va))
            return vl, va, va_base, va_new
        else:
            print('epo {}, test, loss={:.4f} acc={:.4f}'.format(epoch, vl, va))
            return vl, va
