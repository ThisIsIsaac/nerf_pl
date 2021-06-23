import torch
from torch import nn

class MSELoss(nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()
        self.loss = nn.MSELoss(reduction='mean')

    def forward(self, inputs, targets):
        loss = self.loss(inputs['rgb_coarse'], targets["rgbs"])
        if 'rgb_fine' in inputs:
            loss += self.loss(inputs['rgb_fine'], targets["rgbs"])

        return loss

class DispLoss(nn.Module):
    def __init__(self):
        super(DispLoss, self).__init__()
        self.L1_loss = nn.SmoothL1Loss(size_average=True)
        self.MSE_loss = nn.MSELoss(reduction='mean')

    def forward(self, inputs, targets):
        loss = self.MSE_loss(inputs['rgb_coarse'], targets["rgbs"])
        if 'rgb_fine' in inputs:
            loss += self.MSE_loss(inputs['rgb_fine'], targets["rgbs"])

        mask = torch.squeeze(inputs["mask"])
        coarse_disp = inputs["depth_coarse"]
        gt_disp = torch.squeeze(targets["gt_disp"])
        loss += self.L1_loss(coarse_disp[mask], gt_disp[mask])
        if "depth_fine" in inputs:
            fine_disp = inputs["depth_fine"]
            loss += self.L1_loss(fine_disp[mask], gt_disp[mask])

        return loss

loss_dict = {'mse': MSELoss, "disp": DispLoss}
