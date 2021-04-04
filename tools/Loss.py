import torch
import torch.nn as nn
import torch.nn.functional as F


# WGAN Discriminator Loss function
class DLoss(nn.Module):

    def __init__(self):
        super(DLoss, self).__init__()
        pass

    def forward(self, target, preds):
        return -(torch.mean(target) - torch.mean(preds))

# WGAN Generator Loss function
class GLoss(nn.Module):
    
    def __init__(self):
        super(GLoss, self).__init__()
        pass

    def forward(self, preds):
        return -torch.mean(preds)



# WGAN-GP with R1-regularization Discriminator Loss function
class SuperDLoss(DLoss):
    
    def __init__(self, gamma=10):
        super(SuperDLoss, self).__init__()
        self.gamma = gamma

    def forward(self, target, preds):
        loss = super().forward(target, preds)
        return self.gamma / (loss**2)


# WGAN-GP with R1-regularization Generator Loss function
class SuperGLoss(GLoss):

    def __init__(self):
        super(SuperGLoss, self).__init__()
        pass

    def forward(self, preds):
        return super().forward(target, preds)
