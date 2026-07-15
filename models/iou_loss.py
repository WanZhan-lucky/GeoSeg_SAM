import torch
import torch.nn as nn
import torch.nn.functional as F

###################################################################
# ########################## iou loss #############################
# ###################################################################
# class IOU(torch.nn.Module):
#     def __init__(self):
#         super(IOU, self).__init__()
#
#     def _iou(self, pred, target):
#         pred = torch.sigmoid(pred)
#         inter = (pred * target).sum(dim=(2, 3))
#         union = (pred + target).sum(dim=(2, 3)) - inter
#         iou = 1 - (inter / union)
#
#         return iou.mean()
#
#     def forward(self, pred, target):
#         return self._iou(pred, target)
#
class IOU(torch.nn.Module):
    def __init__(self):
        super(IOU, self).__init__()

    def forward(self, pred, target):
        """
        pred: (B, C, H, W) logits
        target: (B, H, W) index mask
        """
        pred = torch.softmax(pred, dim=1)  # 转成概率
        target_onehot = F.one_hot(target, num_classes=pred.shape[1])  # (B,H,W,C)  target = 3  [0,0,0,1,0]  (B, H, W) → (B, H, W, C)

        target_onehot = target_onehot.permute(0,3,1,2).float() #(B, C, H, W)

        inter = (pred * target_onehot).sum(dim=(2,3))       #pred_c * gt_c
        union = (pred + target_onehot).sum(dim=(2,3)) - inter
        iou = inter / (union + 1e-6)
        return 1 - iou.mean()
        # IoU = | A ∩ B | / | A ∪ B |
        # Union = A + B - Intersection

# IoU = |p ∩ g| / |p ∪ g|
# Dice = 2 |p ∩ g| / (|p| + |g|)
class DiceLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super(DiceLoss, self).__init__()
        self.eps = eps

    def forward(self, pred, target):
        """
        pred:   (B, C, H, W) logits
        target: (B, H, W)    int64, 每个像素类别标签 0..C-1
        """
        # 1. 概率
        prob = torch.softmax(pred, dim=1)                  # (B,C,H,W)

        # 2. one-hot GT
        num_classes = pred.shape[1]
        target_onehot = F.one_hot(target, num_classes=num_classes)    # (B,H,W,C)
        target_onehot = target_onehot.permute(0, 3, 1, 2).float()     # (B,C,H,W)

        # 3. 按类计算 Dice：对 batch + H + W 求和
        dims = (0, 2, 3)                                   # over B,H,W
        intersection = (prob * target_onehot).sum(dims)    # (C,)
        union = prob.sum(dims) + target_onehot.sum(dims)   # (C,)

        dice = (2 * intersection + self.eps) / (union + self.eps)  # (C,)
        dice_loss = 1 - dice.mean()                        # 所有类平均

        return dice_loss