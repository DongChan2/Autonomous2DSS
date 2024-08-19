import torch
import numpy as np 
import cv2
from torch.utils.data import Dataset,DataLoader
import random
import glob
from transformers import AutoImageProcessor
from enhanement import FastRetinex,GammaCorrection,Dehaze
from scipy.stats import skew,kurtosis
import pandas as pd 
from functools import partial


def is_dark(img,th=2):
    l,a,b = cv2.split(cv2.cvtColor(img,cv2.COLOR_RGB2LAB))
    data = l.flatten()
    if (skew(data)>=th):
        return True
    return False
def is_foggy(img):
    l,a,b = cv2.split(cv2.cvtColor(img,cv2.COLOR_RGB2LAB))
    data = l.flatten()
    s = skew(data)
    if (s>=1) and (s<2):
        return True
    return False
def is_nonflare(img,th=10):
    l,a,b = cv2.split(cv2.cvtColor(img,cv2.COLOR_RGB2LAB))
    data = l.flatten()
    if (kurtosis(data)>=th):
        return True
    return False

def get_dark_gamma(img):
    l,a,b = cv2.split(cv2.cvtColor(img,cv2.COLOR_RGB2LAB))
    data = l.flatten()
    median = np.median(data)
    if median<=5:
        return 3
    elif median<=20:
        return 2.5
    elif median<=35:
        return 2
    elif median<=50:
        return 1.5
    else:
        return 1.2

def get_foggy_gamma(img):
    l,a,b = cv2.split(cv2.cvtColor(img,cv2.COLOR_RGB2LAB))
    data = l.flatten()
    median = np.median(data)
    if median<=20:
        return 2.5
    elif median<=35:
        return 2
    else:
        return 1.5
    
def do_fastretinex(img):
    l,a,b = cv2.split(cv2.cvtColor(img,cv2.COLOR_RGB2LAB))
    data = l.flatten()
    median = np.median(data)
    if (20 <median) and (median<=50):
        return True
    return False

def collate_fn(batch, image_processor):
    inputs = list(zip(*batch))
    images = inputs[0]
    segmentation_maps = inputs[1]
    batch = image_processor(
        images,
        segmentation_maps=segmentation_maps,
        return_tensors='pt',
    )
    batch['orig_image'] = images
    batch['orig_mask'] = segmentation_maps
    return batch

class CustomDataset(Dataset):
    def __init__(self, root_path, mode, transform=None,do_enhance=False,hybrid_sampling=False):
        super().__init__()
        print("\n---[ Dataset init ]---\n")
        self.root_path = root_path
        self.mode = mode
        self.img_path = sorted(glob.glob(self.root_path+f"/{self.mode}/images/*.jpg"))
        self.mask_path = sorted(glob.glob(self.root_path+f"/{self.mode}/label_img/*.png"))
        self.transform = transform
        self.totensor = torch.LongTensor
        self.do_enhance = do_enhance
        if (mode=='training') and hybrid_sampling:
            names = pd.read_csv(f"{self.root_path}/OverSamplingFileNames.csv")['FileName']
            self.img_path = [f"{self.root_path}/training/images/{i}.jpg" for i in names]
            self.mask_path = [f"{self.root_path}/training/label_img/{i}.png" for i in names]
    def __len__(self):
        return len(self.img_path) 
    
    
    def enhance(self,img):
        """
        1. only gamma correction
        2. fastretinex + gamma correction
        3. dehazing + gamma correction
        gamma correction + fastretinex + dehazing
        """
        if is_dark(img):
            if random.random()<0.5:
                gamma = get_dark_gamma(img)
                if is_nonflare(img):
                    if do_fastretinex(img):
                        return GammaCorrection(FastRetinex(img),gamma)
                else:
                    img = Dehaze(img)
                return GammaCorrection(img,gamma)
        
        elif is_foggy(img):
            if random.random()<0.5:
                gamma = get_foggy_gamma(img)
                if is_nonflare(img):
                    if do_fastretinex(img):
                        return GammaCorrection(FastRetinex(img),gamma)
                else:
                    img = Dehaze(img)
                return GammaCorrection(img,gamma)
        return img
    def __getitem__(self, idx):
        img = cv2.imread(self.img_path[idx])
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.img_path[idx].replace("images",'label_img').replace("jpg","png"),0)

        if self.mode == 'training':
            if self.do_enhance:
                img = self.enhance(img)
            transformed= self.transform['aug'](image = img, mask = mask)
        else:
            transformed= self.transform['origin'](image = img, mask = mask)
            
        return transformed['image'],self.totensor(transformed['mask'])
      
def get_loader(root_path,mode,transform,batch_size,do_enhance=False,hybrid_sampling=False,**kwargs):
    processor = AutoImageProcessor.from_pretrained("facebook/mask2former-swin-small-cityscapes-semantic",ignore_index=23,size={'height':320,'width':480})
    ds = CustomDataset(root_path=root_path, mode=mode,transform = transform,do_enhance=do_enhance,hybrid_sampling=hybrid_sampling)
    if mode =='training':
        return DataLoader(dataset=ds,batch_size=batch_size,pin_memory=True, shuffle= True,num_workers=8,collate_fn=partial(collate_fn, image_processor=processor))    
    return DataLoader(dataset=ds,batch_size=batch_size,pin_memory=True, shuffle= False, num_workers=8,collate_fn=partial(collate_fn, image_processor=processor))