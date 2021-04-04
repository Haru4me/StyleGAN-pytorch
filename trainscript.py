import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn as nn
from PIL import Image
from torchvision.utils import save_image, make_grid
from torch.autograd import Variable

from tqdm import tqdm
import argparse
import random
import os

from tools.Model import StyleGenerator, StyleDiscriminator
from tools.Loss import *
from tools.Dataset import CelebA, FFHQ

import warnings
warnings.filterwarnings("ignore")


parser = argparse.ArgumentParser()

parser.add_argument("--epoch", type=int, default=0, help="Last epoch")
parser.add_argument("--n_epoch", type=int, default=500, help="Epoch number")
parser.add_argument("--batch_size", type=int, default=32, help="Size of the batches")
parser.add_argument("--sample_interval", type=int, default=10, help="Interval between saving generator outputs")
parser.add_argument("--checkpoint_interval", type=int, default=50, help="Interval between saving model checkpoints")
parser.add_argument("--loss", type=str, default='norm', help="Loss func (normal/r1)")
parser.add_argument("--gamma", type=float, default=10.0, help="Loss weight")
parser.add_argument("--seed", type=int, default=7, help="Random seed value")
parser.add_argument("--lr", type=float, default=2e-3, help="Learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="Momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="Momentum of gradient")
parser.add_argument("--weights", type=int, default=1, help="Init weights")
parser.add_argument("--mixing", type=int, default=1, help="Use mixing normalisation")
parser.add_argument("--break_point", type=int, default=2,help="Break-point value")
opt = parser.parse_args()


os.makedirs("./assets/val_imgs", exist_ok=True)
os.makedirs("./assets/saved_models", exist_ok=True)


def samples(num_epoch, gen, real1=None, real2=None):
    
    if real1 is None and real2 is None:
        nrow = min(gen.size(0)//2,8)
        save_image(gen.cpu().data, "./assets/val_imgs/%s.png" %
               num_epoch, nrow=nrow)
    else:
        real1 = torch.cat((torch.ones(real1[0].size()).unsqueeze(0),real1),dim=0)
        real1 = make_grid(real1.cpu().data, nrow=1)
        gen = torch.cat((real2,gen),dim=0)
        gen = make_grid(gen, nrow=real2.size(0))
        gen = torch.cat((real1,gen),dim=2)
        save_image(gen.cpu().data, "./assets/val_imgs/%s.png" % num_epoch)


def train(G, 
          D, 
          optimizer_G, 
          optimizer_D,
          criterion_G, 
          criterion_D, 
          sample_interval=opt.sample_interval, 
          checkpoint_interval=opt.checkpoint_interval,
          num_epoch=opt.n_epoch, 
          lam=1,
          mixing=opt.mixing,
          break_point=opt.break_point,
          batch_size=opt.batch_size):

    print("Star training...")

    pbar = tqdm(range(num_epoch))
    phase = [0, 100, 200, 300, 400, num_epoch]
    step = -1
    size = [(8,8),(16,16),(32,32),(64,64),(128,128)]

    for epoch in pbar:

        run_loss_G = []
        run_loss_D = []

        if epoch == phase[0]:
            cur = phase.pop(0)
            step += 1
            tr = transforms.Compose([transforms.Resize(size[step]),
                                     transforms.ToTensor()])
            train_data = DataLoader(
                CelebA(transform=tr), batch_size=batch_size)

        if phase:
            alpha = 1 - (phase[0] - epoch) / (phase[0]-cur)
        else:
            alpha = 1

        for batch in train_data:

            real = batch.to(device)

            valid = torch.zeros((real.shape[0], 1)).to(device)
            fake = torch.ones((real.shape[0], 1)).to(device)

            if opt.mixing:
                z_in11, z_in12, z_in21, z_in22 = torch.randn(4, 
                        real.size(0), 512, device=device).chunk(4, 0)
                z_tr = [z_in11.squeeze(0), z_in12.squeeze(0)]
                z_val = [z_in21.squeeze(0), z_in22.squeeze(0)]
            else:
                z_in1, z_in2 = torch.randn(2, real.size(
                    0), 512, device=device).chunk(2, 0)
                z_tr = z_in1.squeeze(0)
                z_val = z_in2.squeeze(0)

            """
                Train Generators
            """

            G.requires_grad_(True)
            D.requires_grad_(False)

            G.train()
            optimizer_G.zero_grad()

            pred = G(z_tr, step=step, alpha=alpha, break_point=break_point)
            loss_G = criterion_G(D(pred, step=step, alpha=alpha), valid)

            # Total loss
            loss_G = lam * loss_G

            loss_G.backward()
            optimizer_G.step()

            """
                Train Discriminator
            """

            G.requires_grad_(False)
            D.requires_grad_(True)

            D.train()
            optimizer_D.zero_grad()

            loss_D = 0.5 * (criterion_D(D(real, step=step, alpha=alpha), valid) + \
                            criterion_D(D(pred.detach(), step=step, alpha=alpha), fake))

            loss_D.backward()
            optimizer_D.step()

            run_loss_D.append(loss_D.item())
            run_loss_G.append(loss_G.item())

        """
            Validation
        """

        if not epoch % sample_interval:

            G.eval()

            fake_val = G(z_val, step=step, alpha=alpha)
            samples(epoch, fake_val)

        if not epoch % checkpoint_interval:

            torch.save(G, './Results/saved_model/G.pth')
            torch.save(D, './Results/saved_model/D.pth')

        pbar.set_description("Loss D: %d, Loss G: %d" % \
                             (np.mean(run_loss_D), np.mean(run_loss_G)))

    print("Finished!")


if __name__ == "__main__":

    # Seeds
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    random.seed(opt.seed)

    # Losses
    if opt.loss == 'norm':
        criterion_G = nn.MSELoss()
        criterion_D = nn.MSELoss()
    elif opt.loss == 'r1':
        criterion_G = WGANGP()
        criterion_D = WGANGP()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    G = StyleGenerator()
    D = StyleDiscriminator()

    criterion_G.to(device)
    criterion_D.to(device)

    optimizer_G = torch.optim.Adam(G.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_D = torch.optim.Adam(D.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    train(G, D, optimizer_G, optimizer_D, criterion_G, criterion_D)
