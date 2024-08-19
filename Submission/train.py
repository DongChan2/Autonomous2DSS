import torch
from torch.optim import AdamW
from argparse import ArgumentParser
import dataset
import albumentations as A
import numpy as np 
import random
import trainer
import models
from lr_scheduler import PolynomialLRDecay


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
    transform = {'aug': 
                        A.Compose([ A.RandomResizedCrop(height=320, width=480, scale=(0.75, 1.0), ratio=(1.3,1.6), p=1.0),
                                    A.HorizontalFlip(p=0.5),
                                    A.RandomBrightnessContrast(p=0.5,brightness_limit=0.5,contrast_limit=0),
                                    A.UnsharpMask(p=0.5),
                                ]),
                "origin": A.Compose([ A.NoOp(p=1)] )}
    return transform

def param_filter(name):
    if 'pixel_level_module.encoder.encoder' in name:
        if 'norm' in name:
            return False
        return False
    if 'level_embed' in name:
        return False
    if 'relative_position_bias_table' in name:
        return False
    if 'queries' in name:
        return False
    return True

def main(args):
    seed_everything(2024)  #@@ seed 고정
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    hparam ={'EPOCH':args.epochs,'BATCH_SIZE':args.batch_size,'lr':args.lr,'weight_decay':args.weight_decay,"model":'Mask2Former','memo':'original_split_320x480_with_random_and_retinex_dehaze)_Oversampling'}
    root_path = args.root_path
    transform = get_transform()

    #@@  dataset
    train_loader = dataset.get_loader(root_path=root_path, transform=transform,mode='training',batch_size=hparam['BATCH_SIZE'],hybrid_sampling=True,do_enhance=True)
    valid_loader = dataset.get_loader(root_path=root_path, transform=transform,mode='validation',batch_size=hparam['BATCH_SIZE'])
    test_loader = dataset.get_loader(root_path=root_path, transform=transform,mode='test',batch_size=hparam['BATCH_SIZE'])
   
    loaders=[train_loader,valid_loader]
    model = models.CustomModel(24)
    base_lr = hparam['lr']
    base_weight_decay = hparam['weight_decay']
    param_groups = [
                    {   'params': model.parameters(),
                        'weight_decay': base_weight_decay,
                        'lr': base_lr},
                    {   'params': [param for name, param in model.named_parameters() if ('pixel_level_module.encoder.encoder' in name)],
                        'weight_decay': base_weight_decay * 1.0,
                        'lr': base_lr * 0.1},
                    {   'params': [param for name, param in model.named_parameters() if ('pixel_level_module.encoder.encoder') in name and ('norm' in name)],
                        'weight_decay': base_weight_decay * 0.0,
                        'lr': base_lr * 0.1},
                    {   'params': [param for name, param in model.named_parameters() if 'level_embed' in name or 'queries' in name ],
                        'weight_decay': base_weight_decay * 0.0,
                        'lr': base_lr * 1.0},
                    {   'params': [param for name, param in model.named_parameters() if 'relative_position_bias_table' in name],
                        'weight_decay': base_weight_decay * 0.0,
                        'lr': base_lr * 0.1}
                    ]
    used_params = set()
    unique_param_groups = []

    for group in param_groups:
        unique_params = [p for p in group['params'] if p not in used_params]
        used_params.update(unique_params)
        if unique_params:
            unique_param_groups.append({**group, 'params': unique_params})
    optimizer = AdamW(unique_param_groups)
    lr_scheduler=PolynomialLRDecay(optimizer, max_decay_steps=3938*50, power=0.9, end_learning_rate=0)
    best_model = trainer.train(model,loaders,optimizer,hparam,device,lr_scheduler=lr_scheduler,save_ckpt=True)
    model.load_state_dict(torch.load(best_model)['MODEL'])
    test_miou = trainer.test(model,test_loader,device)


if __name__ =='__main__':
    parser = ArgumentParser()

    parser.add_argument("--epochs", default=50,type = int)
    parser.add_argument("--batch_size", default=4,type = int)
    parser.add_argument("--lr", default=1e-4,type = float)
    parser.add_argument("--weight_decay", default=0.05,type = float)
    parser.add_argument("--root_path",default='./data',type = str)
    args = parser.parse_args()

    main(args=args)