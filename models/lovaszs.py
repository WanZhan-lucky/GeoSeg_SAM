# lovasz_loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F

def lovasz_grad(gt_sorted: torch.Tensor) -> torch.Tensor:
    """
    gt_sorted: (P,) float tensor in {0,1}
    """
    p = gt_sorted.numel()
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.cumsum(0)
    union = gts + (1.0 - gt_sorted).cumsum(0)
    jaccard = 1.0 - intersection / (union + 1e-6)
    if p > 1:
        jaccard[1:] = jaccard[1:] - jaccard[:-1]
    return jaccard

def flatten_probas(probas: torch.Tensor, labels: torch.Tensor, ignore_index=None):
    """
    probas: (B,C,H,W) after softmax
    labels: (B,H,W) int64
    return: probas_flat (P,C), labels_flat (P,)
    """
    B, C, H, W = probas.shape
    probas = probas.permute(0, 2, 3, 1).reshape(-1, C)
    labels = labels.reshape(-1)
    if ignore_index is None:
        return probas, labels
    valid = labels != ignore_index
    return probas[valid], labels[valid]

def lovasz_softmax_flat(probas: torch.Tensor, labels: torch.Tensor, classes="present"):
    """
    probas: (P,C)
    labels: (P,)
    """
    C = probas.size(1)
    losses = []

    if classes in ["all", "present"]:
        class_ids = range(C)
    else:
        class_ids = classes  # list/tuple of class ids

    for c in class_ids:
        fg = (labels == c).float()
        if classes == "present" and fg.sum() == 0:
            continue
        pc = probas[:, c]
        errors = (fg - pc).abs()
        errors_sorted, perm = torch.sort(errors, descending=True)
        fg_sorted = fg[perm]
        grad = lovasz_grad(fg_sorted)
        losses.append(torch.dot(errors_sorted, grad))

    if len(losses) == 0:
        return probas.sum() * 0.0
    return torch.mean(torch.stack(losses))

def lovasz_softmax(logits: torch.Tensor, labels: torch.Tensor, classes="present", per_image=False, ignore_index=None):
    """
    logits: (B,C,H,W) raw logits
    labels: (B,H,W) int64
    """
    if per_image:
        losses = []
        for logit, lab in zip(logits, labels):
            prob = F.softmax(logit.unsqueeze(0), dim=1)
            prob, lab = flatten_probas(prob, lab.unsqueeze(0), ignore_index)
            losses.append(lovasz_softmax_flat(prob, lab, classes=classes))
        return torch.mean(torch.stack(losses))
    else:
        prob = F.softmax(logits, dim=1)
        prob, labels = flatten_probas(prob, labels, ignore_index)
        return lovasz_softmax_flat(prob, labels, classes=classes)

class LovaszSoftmaxLoss(nn.Module):
    def __init__(self, classes="present", per_image=False, ignore_index=None):
        super().__init__()
        self.classes = classes
        self.per_image = per_image
        self.ignore_index = ignore_index

    def forward(self, pred_logits: torch.Tensor, target_index: torch.Tensor) -> torch.Tensor:
        return lovasz_softmax(
            pred_logits,
            target_index,
            classes=self.classes,
            per_image=self.per_image,
            ignore_index=self.ignore_index,
        )
