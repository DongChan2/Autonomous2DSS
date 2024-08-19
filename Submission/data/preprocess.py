import glob
import tqdm 
import json
import numpy as np
import cv2
import pandas as pd
import os

LABEL_MAP={
    'road': 0, 'sidewalk': 1, 'road roughness': 2, 'road boundaries': 3, 'crosswalks': 4, 'lane': 5,
    'road color guide': 6, 'road marking': 7, 'parking': 8, 'traffic sign': 9, 'traffic light': 10,
    'pole/structural object': 11, 'building': 12, 'tunnel': 13, 'bridge': 14, 'pedestrian': 15, 'vehicle': 16,
    'bicycle': 17, 'motorcycle': 18, 'personal mobility': 19, 'dynamic': 20, 'vegetation': 21, 'sky': 22, 'static': 23
}

def json_to_label(json_file):
    parent = "/".join(json_file.split("/")[:-2])
    
    with open(json_file, 'r') as f:
        data = json.load(f)
    label_img = np.zeros(data['image_size'][::-1], dtype=np.uint8)
    for ann in data['Annotation']:
        points=[]
        label = LABEL_MAP[ann['class_name']]
        for x,y in zip(ann['data'][0][::2],ann['data'][0][1::2]):
            points.append((int(x),int(y))) 
        polygon = np.array(points, dtype=np.int32)
        polygon = polygon.reshape((-1, 1, 2))
        cv2.fillPoly(label_img, [polygon], label) 
        name = data['image_name'].split(".")[0]
    os.makedirs(f"{parent}/label_img",exist_ok=True)
    cv2.imwrite(f"{parent}/label_img/{name}.png", (label_img))



if __name__=='__main__':
    paths = glob.glob("./data/*/labels/*.json")
    for path in tqdm.tqdm(paths):
        json_to_label(path)
    print("Make annotations is done!!")

