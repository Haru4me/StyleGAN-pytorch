import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad


# WGAN-GP with R1-regularization Discriminator Loss function
class WGANGP_G(nn.Module):
    
    def __init__(self, penalty=None):
        super(WGANGP_G, self).__init__()

    def forward(self, preds):
        return -preds.mean()


class WGANGP_D(nn.Module):

    def __init__(self, lam=10.0, eps=0.001, penalty_target=1.0, penalty_type='grad'):
        super(WGANGP_D, self).__init__()
        
        self.lam = lam
        self.eps = eps
        self.penalty_type = penalty_type
        self.penalty_target = penalty_target

    def forward(self, real, gen, img):
        
        loss = -real.mean() + gen.mean()

        if self.penalty_type == 'grad':
            grad_real = grad(outputs=real.sum(),
                            inputs=img, create_graph=True)[0]
            penalty = (grad_real.view(grad_real.size(0),
                            -1).norm(2, dim=1) ** 2).mean()
            penalty *= self.lam / self.penalty_target**2
            
        elif self.penalty_type == 'eps':
            penalty = self.eps * (real ** 2).mean()

        return loss + penalty

