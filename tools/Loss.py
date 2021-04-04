import torch
import torch.nn as nn
import torch.nn.functional as F


# WGAN-GP with R1-regularization Discriminator Loss function
class WGANGP(nn.Module):
    
    def __init__(self):
        super(WGANGP, self).__init__()

    def forward(self, preds, target):
        return F.softplus(preds).mean()

