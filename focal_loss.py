import torch
import torch.nn as nn


class FocalLoss(nn.Module):

    def __init__(self, gamma=0):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.ce = torch.nn.CrossEntropyLoss()

    def forward(self, input, target):
        logp = self.ce(input, target)
        p = torch.exp(-logp)
        loss = (1 - p) ** self.gamma * logp
        return loss.mean()

# class FocalLoss(nn.Module):
#     def __init__(self, gamma=0, size_average=True):
#         super(FocalLoss, self).__init__()
#         self.gamma = gamma
#
#     def forward(self, input, target):
#         if input.dim() > 2:
#             input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
#             input = input.transpose(1, 2)  # N,C,H*W => N,H*W,C
#             input = input.contiguous().view(-1, input.size(2))  # N,H*W,C => N*H*W,C
#         target = target.view(-1, 1)
#
#         logpt = F.log_softmax(input)
#         logpt = logpt.gather(1, target)
#         logpt = logpt.view(-1)
#         pt = Variable(logpt.data.exp())
#
#         loss = -1 * (1 - pt) ** self.gamma * logpt
#         return loss.mean()
