# src/focal_loss.py
import torch
import torch.nn.functional as F

class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=0.7, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, reduction='none')
        probs = F.softmax(logits, dim=1)
        pt = probs[torch.arange(len(targets)), targets]
        loss = ((1 - pt) ** self.gamma) * ce
        alpha_vec = torch.where(targets == 1, self.alpha, 1.0 - self.alpha)
        return (alpha_vec * loss).mean()
