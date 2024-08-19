import torch
from argparse import ArgumentParser
import dataset
import albumentations as A
import numpy as np 
import random
import trainer
import models



def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True

def get_transform():
    transform = {"origin": A.Compose([ A.NoOp(p=1)] )}
    return transform

def main(args):
    seed_everything(2024)  #@@ seed 고정
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    root_path = args.root_path
    transform = get_transform()

    #@@  dataset
    test_loader = dataset.get_loader(root_path=root_path, transform=transform,mode='test',batch_size=1)
    model = models.CustomModel(24)
    model.load_state_dict(torch.load(args.model_path,map_location=device)['MODEL'])
    trainer.test(model,test_loader,device)

    

if __name__ =='__main__':
    parser = ArgumentParser()
    parser.add_argument("--root_path",default='./data',type = str)
    parser.add_argument("--model_path",default='./weight/proposed.pth',type = str)
    args = parser.parse_args()

    main(args=args)