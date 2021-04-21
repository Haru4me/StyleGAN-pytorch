import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad


# WGAN-GP with R1-regularization Discriminator Loss function
class WGANGP_G(nn.Module):
    
    def __init__(self, penalty=None):
        super(WGANGP_G, self).__init__()

    def forward(self, preds):
        return F.softplus(-preds).mean()


class WGANGP_D(nn.Module):

    def __init__(self, lam=10.0, eps=0.001, penalty_target=1.0, penalty_type='grad'):
        super(WGANGP_D, self).__init__()
        
        self.lam = lam
        self.eps = eps
        self.penalty_type = penalty_type
        self.penalty_target = penalty_target

    def forward(self, real, gen, img):
        
        loss = F.softplus(-real).mean() + F.softplus(gen).mean()

        if self.penalty_type == 'grad':
            grad_real = grad(outputs=real.sum(),
                            inputs=img, create_graph=True)[0]
            penalty = (grad_real.view(grad_real.size(0),
                            -1).norm(2, dim=1) ** 2).mean()
            penalty *= self.lam / self.penalty_target**2
            
        elif self.penalty_type == 'eps':
            penalty = self.eps * (real ** 2).mean()

        return loss + penalty


class MSE_G(nn.Module):

    def __init__(self, reduction='mean'):
        super(MSE_G, self).__init__()
        self.reduction = reduction

    def forward(self, preds):
        return F.mse_loss(preds, torch.ones(preds.size(), dtype=preds.dtype,
                                            device=preds.device), reduction=self.reduction)


class MSE_D(nn.Module):

    def __init__(self, lam=10.0, eps=0.001, penalty_target=1.0,
                 penalty_type='grad', reduction='mean'):
        super(MSE_D, self).__init__()

        self.lam = lam
        self.eps = eps
        self.penalty_type = penalty_type
        self.penalty_target = penalty_target
        self.reduction = reduction

    def forward(self, real, gen, img):

        loss = 0.5*(F.mse_loss(real, torch.ones(real.size(), dtype=real.dtype, device=real.device),
                               reduction=self.reduction) +
                    F.mse_loss(gen, torch.zeros(gen.size(), dtype=gen.dtype, device=gen.device),
                               reduction=self.reduction))

        if self.penalty_type == 'grad':
            grad_real = grad(outputs=real.sum(),
                             inputs=img, create_graph=True)[0]
            penalty = (grad_real.view(grad_real.size(0),
                                      -1).norm(2, dim=1) ** 2).mean()
            penalty *= self.lam / self.penalty_target**2
            loss += penalty

        elif self.penalty_type == 'eps':
            penalty = self.eps * (real ** 2).mean()
            loss += penalty

        return loss
