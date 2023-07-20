import yaml
with open('config.yaml') as fh:
    config = yaml.load(fh, Loader=yaml.FullLoader)
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch
import numpy as np
try:
    from itertools import  ifilterfalse
except ImportError: # py3k
    from itertools import  filterfalse as ifilterfalse



class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super(BinaryFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        return F_loss.mean()
#%%

def one_hot_encode(target, num_classes):
    """
    Converts `target` to one-hot encoding.

    Parameters:
    - target: a tensor of shape (batch_size, 1, height, width).
    - num_classes: the number of classes.

    Returns:
    - A tensor of shape (batch_size, num_classes, height, width).
    """
    one_hot = torch.zeros(target.size(0), num_classes, target.size(2), target.size(3)).to(target.device)
    target_adjusted = target.long()
    return one_hot.scatter_(1, target_adjusted, 1)

class DiceLoss(nn.Module):
    def __init__(self, smooth=1., one_hot=True):
        super(DiceLoss, self).__init__()
        self.one_hot = one_hot
        self.smooth = smooth

    def forward(self, preds, targets):
        preds = F.softmax(preds, dim=1)  # Apply softmax to get probabilities

        if self.one_hot:
            targets = one_hot_encode(targets, preds.size(1))
        targets = targets.type_as(preds)  # Ensure the targets are of the same type as the predictions

        # Calculate the Dice coefficient
        intersection = (preds * targets).sum(dim=(2, 3))
        dice = (2. * intersection + self.smooth) / (preds.sum(dim=(2, 3)) + targets.sum(dim=(2, 3)) + self.smooth)

        # Return the Dice loss
        return 1. - dice.mean()

class FocalTverskyLoss(torch.nn.Module):
    def __init__(self, alpha=0.2, beta=0.8, gamma=2.0, smooth=1e-5, one_hot=True):
        super(FocalTverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth
        self.one_hot = one_hot

    def forward(self, inputs, targets):
        # Compute the probabilities
        inputs = F.softmax(inputs, dim=1)
        if self.one_hot:
            targets = one_hot_encode(targets, inputs.size(1))
        # Compute the Tversky index
        tp = torch.sum(inputs * targets, dim=(2, 3))
        fp = torch.sum(inputs * (1 - targets), dim=(2, 3))
        fn = torch.sum((1 - inputs) * targets, dim=(2, 3))

        tversky_index = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)

        # Compute the Focal Tversky loss
        loss = torch.pow((1 - tversky_index), self.gamma)

        return loss.mean()

#%%
def one_hot(index, classes):
    
    size = index.size()[:1] + (classes,) # jsut gets shape -> (P, C)
    view = index.size()[:1] + (1,) # get shapes -> (P, 1)
    
    # makes a tensor of size (P, C) and fills it with zeros
    mask = torch.Tensor(size).fill_(0).to('cuda' if torch.cuda.is_available() else 'cpu') # (P, C)
    # reshapes the targets/index to (P, 1)
    index = index.view(view) 
    ones = 1.

    return mask.scatter_(1, index, ones) # places ones at those indexes (in mask) indicated by values in targets 


class FocalLoss(nn.Module):

    def __init__(self, gamma=2, eps=1e-7, size_average=True, one_hot=True, ignore=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps
        self.size_average = size_average
        self.one_hot = one_hot
        self.ignore = ignore

    def forward(self, input, target):
        '''
        0th index is ignored.
        '''
        B, C, H, W = input.size()
        input = input.permute(0, 2, 3, 1).contiguous().view(-1, C)  # B * H * W, C => (P, C) 
        target = target.view(-1) # (P,)
        
        if self.ignore is not None:
            valid = (target != self.ignore)
            input = input[valid]
            target = target[valid]

        if self.one_hot:
            target = one_hot(target, C) # (P, C)

        probs = F.softmax(input, dim=1)
        probs = (probs * target).sum(1)
        probs = probs.clamp(self.eps, 1. - self.eps)

        log_p = probs.log()

        batch_loss = -(torch.pow((1 - probs), self.gamma)) * log_p

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss
#%%

#PyTorch
# ALPHA  < 0.5 penalises FP more, > 0.5 penalises FN more
# CE_RATIO = 0.5 #weighted contribution of modified CE loss compared to Dice loss

class ComboLoss(nn.Module):
    def __init__(self, smooth=1, beta=0.3, alpha=0.5,
                 weight=None, size_average=True):
        super(ComboLoss, self).__init__()
        self.smooth = smooth
        self.alpha = alpha
        self.one_hot = True
        self.beta = beta

    def forward(self, inputs, targets):
        e = 1e-7 # epsilon for stability
        
        #flatten label and prediction tensors
        inputs = F.softmax(inputs, dim=1)
        inputs = inputs.permute(0, 2, 3, 1).contiguous().view(-1, inputs.size(1))  # B * H * W, C => (P, C) 

        targets = targets.view(-1) # (P,)
        if self.one_hot:
            targets = one_hot(targets, inputs.size(1)) # (P, C)
        #True Positives, False Positives & False Negatives
        intersection = (inputs * targets).sum()    
        dice = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)
        
        inputs = torch.clamp(inputs, e, 1.0 - e)       
        out = - (self.beta * ((targets * torch.log(inputs)) + ((1 - self.beta) * (1.0 - targets) * torch.log(1.0 - inputs))))
        weighted_ce = out.mean(-1)
        combo = (self.alpha * weighted_ce) - ((1 - self.alpha) * dice)
        
        return combo.mean()

# --------------------------- BINARY LOSSES ---------------------------


def lovasz_hinge(logits, labels, per_image=True, ignore=None):
    """
    Binary Lovasz hinge loss
      logits: [B, H, W] Variable, logits at each pixel (between -\infty and +\infty)
      labels: [B, H, W] Tensor, binary ground truth masks (0 or 1)
      per_image: compute the loss per image instead of per batch
      ignore: void class id
    """
    # logits = logits.squeeze(1)
    # labels = labels.squeeze(1)
    
    if per_image:
        loss = mean(lovasz_hinge_flat(*flatten_binary_scores(log.unsqueeze(0), lab.unsqueeze(0), ignore))
                          for log, lab in zip(logits, labels))
    else:
        loss = lovasz_hinge_flat(*flatten_binary_scores(logits, labels, ignore))
    return loss


def lovasz_hinge_flat(logits, labels):
    """
    Binary Lovasz hinge loss
      logits: [P] Variable, logits at each prediction (between -\infty and +\infty)
      labels: [P] Tensor, binary ground truth labels (0 or 1)
      ignore: label to ignore
    """
    if len(labels) == 0:
        # only void pixels, the gradients should be 0
        return logits.sum() * 0.
    signs = 2. * labels.float() - 1.
    errors = (1. - logits * Variable(signs))
    errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
    perm = perm.data
    gt_sorted = labels[perm]
    grad = lovasz_grad(gt_sorted)
    loss = torch.dot(F.relu(errors_sorted), Variable(grad))
    return loss


def flatten_binary_scores(scores, labels, ignore=None):
    """
    Flattens predictions in the batch (binary case)
    Remove labels equal to 'ignore'
    """
    scores = scores.view(-1)
    labels = labels.view(-1)
    if ignore is None:
        return scores, labels
    valid = (labels != ignore)
    vscores = scores[valid]
    vlabels = labels[valid]
    return vscores, vlabels

def isnan(x):
    return x != x

def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    if p > 1: # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard

def mean(l, ignore_nan=False, empty=0):
    """
    nanmean compatible with generators.
    """
    l = iter(l)
    if ignore_nan:
        l = ifilterfalse(isnan, l)
    try:
        n = 1
        acc = next(l)
    except StopIteration:
        if empty == 'raise':
            raise ValueError('Empty mean')
        return empty
    for n, v in enumerate(l, 2):
        acc += v
    if n == 1:
        return acc
    return acc / n


class LovaszSoftmax(nn.Module):
    def __init__(self, classes='present', per_image=False, ignore=None):
        super(LovaszSoftmax, self).__init__()
        self.classes = classes
        self.per_image = per_image
        self.ignore = ignore

    def forward(self, inputs, targets):
        probas = F.softmax(inputs, dim=1) # B*C*H*W -> from logits to probabilities
        return lovasz_softmax(probas, targets, self.classes, self.per_image, self.ignore)

def lovasz_softmax(probas, labels, classes='present', per_image=False, ignore=None):
    """
    Multi-class Lovasz-Softmax loss
      probas: [B, C, H, W] Variable, class probabilities at each prediction (between 0 and 1).
              Interpreted as binary (sigmoid) output with outputs of size [B, H, W].
      labels: [B, H, W] Tensor, ground truth labels (between 0 and C - 1)
      classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
      per_image: compute the loss per image instead of per batch
      ignore: void class labels
    """
    if per_image:
        loss = mean(lovasz_softmax_flat(*flatten_probas(prob.unsqueeze(0), lab.unsqueeze(0), ignore), classes=classes)
                          for prob, lab in zip(probas, labels))
    else:
        loss = lovasz_softmax_flat(*flatten_probas(probas, labels, ignore), classes=classes)
    return loss


def lovasz_softmax_flat(probas, labels, classes='present'):
    """
    Multi-class Lovasz-Softmax loss
      probas: [P, C] Variable, class probabilities at each prediction (between 0 and 1)
      labels: [P] Tensor, ground truth labels (between 0 and C - 1)
      classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
    """
    if probas.numel() == 0:
        # only void pixels, the gradients should be 0
        return probas * 0.
    C = probas.size(1)
    losses = []
    class_to_sum = list(range(C)) if classes in ['all', 'present'] else classes
    for c in class_to_sum:
        fg = (labels == c).float() # foreground for class c
        if (classes is 'present' and fg.sum() == 0):
            continue
        if C == 1:
            if len(classes) > 1:
                raise ValueError('Sigmoid output possible only with 1 class')
            class_pred = probas[:, 0]
        else:
            class_pred = probas[:, c]
        errors = (Variable(fg) - class_pred).abs()
        errors_sorted, perm = torch.sort(errors, 0, descending=True)
        perm = perm.data
        fg_sorted = fg[perm]
        losses.append(torch.dot(errors_sorted, Variable(lovasz_grad(fg_sorted))))
    return mean(losses)


def flatten_probas(probas, labels, ignore=None):
    """
    Flattens predictions in the batch
    """
    if probas.dim() == 3:
        # assumes output of a sigmoid layer
        B, H, W = probas.size()
        probas = probas.view(B, 1, H, W)
    B, C, H, W = probas.size()
    probas = probas.permute(0, 2, 3, 1).contiguous().view(-1, C)  # B * H * W, C = P, C
    labels = labels.view(-1)
    if ignore is None:
        return probas, labels
    valid = (labels != ignore)
    vprobas = probas[valid.nonzero().squeeze()]
    vlabels = labels[valid]
    return vprobas, vlabels

def xloss(logits, labels, ignore=None):
    """
    Cross entropy loss
    """
    return F.cross_entropy(logits, Variable(labels), ignore_index=255)


# --------------------------- HELPER FUNCTIONS ---------------------------
def isnan(x):
    return x != x
    
    
def mean(l, ignore_nan=False, empty=0):
    """
    nanmean compatible with generators.
    """
    l = iter(l)
    if ignore_nan:
        l = ifilterfalse(isnan, l)
    try:
        n = 1
        acc = next(l)
    except StopIteration:
        if empty == 'raise':
            raise ValueError('Empty mean')
        return empty
    for n, v in enumerate(l, 2):
        acc += v
    if n == 1:
        return acc
    return acc / n


def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    if p > 1: # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard