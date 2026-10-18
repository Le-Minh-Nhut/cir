import torch
import torch.nn as nn
import torch.nn.functional as F


class ComputeFinalLoss(nn.Module):
    def __init__(self, tau):
        super(ComputeFinalLoss, self).__init__()
        self.infonce_loss = infoNCELoss(tau=tau)

    def forward(self, sim_local_1, sim_local_2):
        loss_local_infoNCE_1 = self.infonce_loss(sim_local_1.max(-1)[0])
        loss_local_infoNCE_2 = self.infonce_loss(sim_local_2.max(-1)[0])
        loss_all = loss_local_infoNCE_1 + loss_local_infoNCE_2
        return loss_all


class infoNCELoss(nn.Module):
    def __init__(self, tau):
        super(infoNCELoss, self).__init__()
        self.tau = tau
        self.temp = nn.Parameter(0.07 * torch.ones([]))

    def forward(self, scores):
        bs = scores.size(0)
        scores = scores / self.temp
        targets = torch.linspace(0, bs - 1, bs, dtype=int).to(scores.device)
        scores = scores.permute(0, 1)
        return F.cross_entropy(scores, targets)
