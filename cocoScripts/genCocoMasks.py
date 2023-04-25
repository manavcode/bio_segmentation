import os
import json
import numpy as np
import cv2
from detectron2.data import DatasetCatalog
from detectron2.data.datasets import register_coco_instances

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from pycocotools.coco import COCO

#Load in path to images and annotations
image_dir = '/path/to/livecell_test_images'
anno_file = '/path/to/testImgs.json'

#Load in liveCell COCO file
coco = COCO(anno_file)

#Create new COCO file for predicted masks
new_coco = {}
new_coco['info'] = coco.dataset['info']
new_coco['licenses'] = coco.dataset['licenses']
new_coco['categories'] = coco.dataset['categories']
new_coco['annotations'] = []
new_coco['images'] = []

#Register liveCell COCO instance with path to images and annotations
register_coco_instances('my_dataset', {}, anno_file, image_dir)

# get the test dataset
dataset_dicts = DatasetCatalog.get("my_dataset")

#Load in your model checkpoint for predicting masks (this block of code could be very different for your model)
sam_checkpoint = "/home/MBronars/workspace/segment-anything/segment_data/sam_vit_h_4b8939.pth"
model_type = "vit_h"
device = "cuda"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
mask_generator = SamAutomaticMaskGenerator(sam)
mask_generator.output_mode = "coco_rle"


#Keep track of unique annotation IDs
ann_id = 0

#Make predictions for each image in the test set
for i, d in enumerate(dataset_dicts):
    # read the image
    img = cv2.imread(d["file_name"])
    
    # predict masks (should be a list of dicts with keys: segmentation, area, bbox, stability_score)
    # segmentation should be in COCO RLE format
    # stability score not required, can be any measure of confidence or just leave black and comment out line 74
    masks = mask_generator.generate(img)
    
    #Create new image entry for new COCO file
    new_image = {
        'id': d['image_id'],
        'width': d['width'],
        'height': d['height'],
        'file_name': d['file_name'],
    }
    new_coco['images'].append(new_image)
    
    #Create new annotation entry for each predicted mask
    for m in masks:
        new_ann = {
            'image_id': d['image_id'],
            'id': ann_id,
            'category_id': 1,
            'segmentation': m['segmentation'],
            'area': m['area'],
            'bbox': m['bbox'],
            'iscrowd': 0,
            'score': m['stability_score']
        }
        new_coco['annotations'].append(new_ann)
        ann_id += 1

#Save new COCO file
with open('segResult.json', 'w') as f:
    json.dump(new_coco, f)