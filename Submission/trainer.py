import os
import torch
from tqdm import tqdm 
from datetime import datetime
import numpy as np 
import cv2 


from torch.utils.tensorboard import SummaryWriter
from transformers import AutoImageProcessor
from metrics import StreamSegMetrics



LABEL_MAP = {
    'road': 0, 'sidewalk': 1, 'road roughness': 2,
    'road boundaries': 3, 'crosswalks': 4, 'lane': 5,
    'road color guide': 6, 'road marking': 7, 'parking': 8,
    'traffic sign': 9, 'traffic light': 10, 'pole/structural object': 11,
    'building': 12, 'tunnel': 13, 'bridge': 14, 'pedestrian': 15,
    'vehicle': 16, 'bicycle': 17, 'motorcycle': 18,
    'personal mobility': 19, 'dynamic': 20, 'vegetation': 21,
    'sky': 22, 'static': 23
}

SAVE_PATH = './storage/'


def save_checkpoint(epoch,model,optimizer,miou,name,scheduler=None):
    new_path= SAVE_PATH +name
    os.makedirs(new_path,exist_ok=True)
    ckpt={"MODEL":model.state_dict(),
          "OPTIMIZER":optimizer.state_dict(),
          'EPOCH':epoch,
          "NAME":name}
    if scheduler is not None:
        ckpt.update({"SCHEDULER_STATE":scheduler.state_dict()})
  
    torch.save(ckpt,new_path+f"/Epoch_{epoch},mIOU_{miou*100:.3f}.pth.tar")
    print(f"Model Saved,when Epoch:{epoch}")


def _train_one_step(model,data,optimizer,device,criterion,logger,**kwargs):

    pixel_values = data['pixel_values'].to(device)
    mask_labels = [mask_label.to(device) for mask_label in data['mask_labels']]
    class_labels = [class_label.to(device) for class_label in data['class_labels']]
    pixel_mask = data['pixel_mask'].to(device)
    targets = [target.to(device) for target in data['orig_mask']]
    inputs = {'pixel_values':pixel_values,'mask_labels':mask_labels,'class_labels':class_labels,'pixel_mask':pixel_mask}
    output = model(inputs)
    optimizer.zero_grad()
    loss = output.loss
    target_sizes = [image.shape for image in data['orig_mask']]
    pred_maps = torch.stack(criterion['processor'].post_process_semantic_segmentation(output, target_sizes=target_sizes),dim=0)
    targets=torch.stack(targets,dim=0)
    criterion['metric'].update(targets.detach().cpu().numpy(),pred_maps.detach().cpu().numpy())
    logger.add_scalar("loss/step",loss.item(),kwargs['iter'])  #@@@
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.01)
    optimizer.step()
    
    return {'loss':loss.item()}
    

def _train_one_epoch(model,dataloader,optimizer,device,criterion,logger,**kwargs):

    model.train()
    total_loss = 0
    for batch_index,data in enumerate(tqdm(dataloader)):
        history = _train_one_step(model,data,optimizer,device,criterion,logger,iter=(len(dataloader)*(kwargs['epoch_index'])+(batch_index)))
        total_loss += history['loss']
        kwargs['lr_scheduler'].step()
    results = criterion['metric'].get_results()
    return {'loss':total_loss,'metrics':results}


def _validate_one_step(model,data,device,*args,**kwargs):
    logger = kwargs['logger']
    criterion = kwargs['criterion']
    pixel_values = data['pixel_values'].to(device)
    mask_labels = [mask_label.to(device) for mask_label in data['mask_labels']]
    class_labels = [class_label.to(device) for class_label in data['class_labels']]
    pixel_mask = data['pixel_mask'].to(device)
    targets = [target.to(device) for target in data['orig_mask']]
    inputs = {'pixel_values':pixel_values,'mask_labels':mask_labels,'class_labels':class_labels,'pixel_mask':pixel_mask}
    
    with torch.no_grad():
        output = model(inputs)
    loss = output.loss
    target_sizes = [image.shape for image in data['orig_mask']]
    pred_maps = torch.stack(criterion['processor'].post_process_semantic_segmentation(output, target_sizes=target_sizes),dim=0)
    targets=torch.stack(targets,dim=0)
    criterion['metric'].update(targets.detach().cpu().numpy(),pred_maps.detach().cpu().numpy())
    logger.add_scalar("loss/step",loss.item(),kwargs['iter'])  
    
    return {'loss':loss.item()}
    

def _validate_one_epoch(model,dataloader,device,**kwargs):
    model.eval()
    total_loss = 0
    criterion = kwargs['criterion']
    for batch_index,data in enumerate(tqdm(dataloader)):
        with torch.no_grad():
            history = _validate_one_step(model,data,device,logger=kwargs['logger'],iter=(len(dataloader)*(kwargs['epoch_index'])+(batch_index)), criterion = kwargs['criterion'])
        total_loss += history['loss']
    metrics = criterion['metric'].get_results()
    
    return {'loss':total_loss,'metrics':metrics}


def _test_one_step(model,data,device,*args,**kwargs):
    criterion = kwargs['criterion']
    pixel_values = data['pixel_values'].to(device)
    mask_labels = [mask_label.to(device) for mask_label in data['mask_labels']]
    class_labels = [class_label.to(device) for class_label in data['class_labels']]
    pixel_mask = data['pixel_mask'].to(device)
    targets = [target.to(device) for target in data['orig_mask']]
    inputs = {'pixel_values':pixel_values,'mask_labels':mask_labels,'class_labels':class_labels,'pixel_mask':pixel_mask}
    with torch.no_grad():
        output = model(inputs)
    loss = output.loss
    target_sizes = [image.shape for image in data['orig_mask']]
    pred_maps = torch.stack(criterion['processor'].post_process_semantic_segmentation(output, target_sizes=target_sizes),dim=0)
    targets=torch.stack(targets,dim=0)
    criterion['metric'].update(targets.detach().cpu().numpy(),pred_maps.detach().cpu().numpy())
    
    return {'loss':loss.item()}

def _test_one_epoch(model,dataloader,device,**kwargs):
    model.eval()
    total_loss = 0
    criterion = kwargs['criterion']
    for data in tqdm(dataloader):
        with torch.no_grad():
            history = _test_one_step(model,data,device,criterion = criterion)
        total_loss += history['loss']
    metrics = criterion['metric'].get_results()
    
    return {'loss':total_loss,'metrics':metrics}

def check_infer_time(model,dataloader,device):
    import time
    processor = AutoImageProcessor.from_pretrained("facebook/mask2former-swin-small-cityscapes-semantic",ignore_index=23)
    model.eval()
    model.to(device)
    with torch.no_grad():
        t1 = time.time()
        for data in tqdm(dataloader):
            pixel_values = data['pixel_values'].to(device)
            inputs = {'pixel_values':pixel_values}
            output = model(inputs)
            prediciton = processor.post_process_semantic_segmentation(output, target_sizes=[(1920,1200)])
        t2 = time.time()  
    print(f'Total inference time : {(t2-t1):.2f}s')
               

    

def train(model,loaders,optimizer,hparam,device,lr_scheduler=None,save_ckpt=True):
    dataloader,valid_dataloader = loaders
    print(f"Training Start")
    print("="*200)
    name_list=[]
    for k,v in hparam.items():
        name_list.append(k + ";"+ str(v))
    
    t= datetime.today().strftime("%m_%d_%H;%M;%S")
    name =",".join(name_list)+f"/{t}"
    
    train_logger = SummaryWriter(log_dir = f"./storage/logs/{name}/train")
    valid_logger = SummaryWriter(log_dir = f"./storage/logs/{name}/validation")
    
    model.to(device)
    
    epochs = hparam['EPOCH']
    best_miou =0
    best_model =""
    criterion = {'processor':AutoImageProcessor.from_pretrained("facebook/mask2former-swin-small-cityscapes-semantic",ignore_index=23),
                 'metric':StreamSegMetrics(n_classes=24,ignore_index=23,class_info=LABEL_MAP)}
    for idx,epoch in (enumerate(range(epochs))):
       
        print(f"\rEpoch :{idx+1}/{epochs}")
        history = _train_one_epoch(model,dataloader,optimizer,device,epoch_index=idx,logger=train_logger,criterion=criterion,lr_scheduler=lr_scheduler)
        epoch_loss = history['loss'] / len(dataloader)
        epoch_miou = history['metrics']['Mean IoU']
        train_logger.add_scalar("loss/epoch",epoch_loss,idx)
        train_logger.add_scalar("miou/epoch",epoch_miou,idx)
        print("="*200)
        print(f"Traing {idx+1}/{epochs} Epoch end")
        print("="*200)
        print(criterion['metric'].to_str(history['metrics']))
        print(f"loss:{epoch_loss},  miou:{epoch_miou}")
        history.clear()
        criterion['metric'].reset()
        

        
        valid_history = _validate_one_epoch(model,valid_dataloader,device,epoch_index=idx,logger=valid_logger,criterion=criterion)
        epoch_valid_loss = valid_history['loss'] / len(valid_dataloader)
        epoch_valid_miou = valid_history['metrics']['Mean IoU'] 
        print("="*200)
        print(f"Validation {idx+1}/{epochs} Epoch end")
        print("="*200)
        
        valid_logger.add_scalar("loss/epoch",epoch_valid_loss,idx)
        valid_logger.add_scalar("miou/epoch",epoch_valid_miou,idx)
        print(criterion['metric'].to_str(valid_history['metrics']))
        print(f"validation_loss:{epoch_valid_loss}, validation_miou:{epoch_valid_miou}")
        valid_history.clear()
        criterion['metric'].reset()
        if save_ckpt:
            if best_miou<epoch_valid_miou:
                best_miou = epoch_valid_miou
                best_model = SAVE_PATH +name+"best_miou"+f"/Epoch_{epoch},mIOU_{best_miou*100:.3f}.pth.tar"
                save_checkpoint(epoch,model,optimizer,epoch_valid_miou,name=name+"best_miou")
        torch.cuda.empty_cache()

    train_logger.close()
    valid_logger.close()
    print('Training End')
    print("="*200)
    return best_model

def test(model,loader,device):
    criterion = {'processor':AutoImageProcessor.from_pretrained("facebook/mask2former-swin-small-cityscapes-semantic",ignore_index=23),
                 'metric':StreamSegMetrics(n_classes=24,ignore_index=23,class_info=LABEL_MAP)}
    criterion['metric'].reset()
    model.to(device)
    test_history = _test_one_epoch(model,loader,device,criterion=criterion)
    test_loss = test_history['loss'] / len(loader)
    test_miou = test_history['metrics']['Mean IoU'] 
    print(criterion['metric'].to_str(test_history['metrics']))
    print(f"Test_loss:{test_loss}, Test_miou:{test_miou}")
    return test_miou

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)