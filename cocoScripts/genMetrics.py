from pycocotools import mask as maskUtils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

import numpy as np

# Load the ground truth annotations and predicted results in COCO format
coco_gt = COCO('/path/to/groundTruch/testImgs.json')
coco_results = COCO('/path/to/segResult.json')

# Initialize COCO evaluation object for instance segmentation
coco_eval = COCOeval(coco_gt, coco_results, 'segm')

# Calculate IoU for each image and each object instance and run evaluation metrics
coco_eval.evaluate()
coco_eval.accumulate()
coco_eval.summarize()

#Calculate average IoU
mean_iou = 0
#Loop over all images
for v in coco_eval.ious.values():
    #Grab IoU for each predicted object and calculate the mean
    mean_iou += np.mean(np.max(v, axis=0))

#mean over all images   
mean_iou /= len(coco_eval.ious)

print("Average IoU:", mean_iou)