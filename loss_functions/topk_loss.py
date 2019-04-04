import numpy as np
import torch
from loss_functions.ND_Crossentropy import CrossentropyND


class TopKLoss(CrossentropyND):
    """
    Network has to have NO LINEARITY!
    """

    def __init__(self, weight=None, ignore_index=-100, k=10):
        self.k = k
        super(TopKLoss, self).__init__(weight, False, ignore_index, reduce=False)

    def forward(self, inp, target):
        target = target[:, 0].long()
        res = super(TopKLoss, self).forward(inp, target)
        num_voxels = np.prod(res.shape)
        res, _ = torch.topk(res.view((-1,)), int(num_voxels // self.k), sorted=False)
        return res.mean()