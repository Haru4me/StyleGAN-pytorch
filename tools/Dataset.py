import torch
from torch.utils.data import Dataset
from torchvision import transforms
from pathlib import Path
from PIL import Image
import numpy as np
import pandas as pd



"""
    Dataset for CelebA loading
"""

class CelebA(Dataset):
    
    def __init__(self, root='./data', transform=None):
    
        df = pd.read_csv('{}/list_eval_partition.csv'.format(root))

        index = np.random.permutation(len(df))

        self.root = root
        self.pathes = df.image_id.to_numpy()[index]
        self.transform = transform if transform != None else transforms.ToTensor()

    def __getitem__(self,index):
        
        img = Image.open('{}/img_align_celeba/img_align_celeba/{}'.format(self.root,self.pathes[index]))
        return self.transform(img)

    def __len__(self):
        return self.pathes.__len__()




"""
    Dataset for FFHQ loading
"""

class FFHQ(Dataset):
    
    def __init__(self, root='./data', transform=None):
        
        self.pathes = np.array(list(Path('{}/thumbnails128x128'.format(root)).glob('*.png')))
        index = np.random.permutation(len(self.pathes))

        self.pathes = self.pathes[index]
        self.transform = transform if transform != None else transforms.ToTensor()

    def __getitem__(self, index):
        img = Image.open(self.pathes[index])
        return self.transform(img)

    def __len__(self):
        return self.pathes.__len__()
