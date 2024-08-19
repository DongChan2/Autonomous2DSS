import torch
from argparse import ArgumentParser
import dataset
import albumentations as A
import numpy as np 
import random
import trainer
import models
import cv2 
from transformers import AutoImageProcessor
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


class_dict = {'road': (128, 128, 128),            
    'sidewalk': (192, 192, 192),        
    'road roughness': (64, 64, 64),    
    'road boundaries': (128, 128, 0),  
    'crosswalks': (255, 255, 255),    
    'lane': (0, 255, 255),              
    'road color guide': (0, 255, 0),   
    'road marking': (255, 255, 255),    
    'parking': (255, 0, 0),           
    'traffic sign': (0, 0, 128),      
    'traffic light': (255, 255, 0),   
    'pole/structural object': (128, 128, 0), 
    'building': (160, 82, 45),        
    'tunnel': (180, 130, 70),         
    'bridge': (144, 128, 112),       
    'pedestrian': (255, 192, 203),      
    'vehicle': (0, 0, 255),             
    'bicycle': (255, 0, 255),          
    'motorcycle': (255, 165, 0),     
    'personal mobility': (255, 20, 147),
    'dynamic': (128, 0, 128),           
    'vegetation': (0, 128, 0),        
    'sky': (250, 206, 135),           
    'static':(0,0,0)}
classes = list(class_dict.keys())
palette = list(class_dict.values())

def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True


def infer(model,data_path):
    model.eval()
    img = cv2.imread(data_path)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    processor = AutoImageProcessor.from_pretrained("facebook/mask2former-swin-small-cityscapes-semantic",
                                                   ignore_index=23,size={'height':320,'width':480})
    inputs = processor(img, return_tensors="pt")
    with torch.no_grad():
        outputs = model(inputs)
    prediction = processor.post_process_semantic_segmentation(outputs,target_sizes=[img.shape[:2]])[0]
    
    return prediction    

def make_color_map(img): 
    height, width = img.shape
    colored_img = np.zeros((height, width, 3), dtype=np.uint8)
    for i in range(height):
        for j in range(width):
            pixel_value = img[i, j]
            colored_img[i, j] = list(palette)[pixel_value]
    return colored_img
   
def draw_result(prediction):
    
    fig,ax = plt.subplots(figsize=(20, 10))
    ax.imshow(prediction)
    ax.axis('off')
    patches = [mpatches.Patch(color=np.array(palette[i])/255., 
                            label=classes[i]) for i in range(24)]
    ax.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., 
            fontsize='large')
    fig.show()
    return fig
    
    
def main(args):
    seed_everything(2024)  #@@ seed 고정
    device='cpu'
    data_path = args.data_path

    model = models.CustomModel(24)
    model.to(device)
    model.load_state_dict(torch.load(args.model_path,map_location=device)['MODEL'])
    output = infer(model,data_path)
    prediction = make_color_map(output.detach().cpu().numpy())
    result = draw_result(prediction)
    name = data_path.split('/')[-1].split('.')[0]
    result.savefig(f'{args.save_path}/{name}.png')





if __name__ =='__main__':
    parser = ArgumentParser()
    parser.add_argument("--data_path",default='./data/sample_imgs/SAMPLE.jpg',type = str)
    parser.add_argument("--model_path",default='./weight/proposed.pth',type = str)
    parser.add_argument("--save_path",default='./data/sample_predictions',type = str)
    args = parser.parse_args()

    main(args=args)