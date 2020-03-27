import torch

class MSE_L1_loss(nn.Module):
    def __init__(self, aggregate="sum"):
        super(MSE_L1_loss, self).__init__()
        self.aggregate = aggregate
        self.MSE = nn.MSELoss()
        self.L1 = nn.L1Loss()
    def forward(self, net_output, target):
        MSE_loss = self.MSE(net_output, target)
        L1_loss = self.L1(net_output, target)
        if self.aggregate == "sum":
            result = 0.3 * MSE_loss + 0.7 * L1_loss
        else:
            raise NotImplementedError("nah son") # reserved for other stuff (later)
        return result
