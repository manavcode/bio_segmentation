import torch
import torch.nn as nn
import detectron2.modeling as modeling
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer
from detectron2.modeling import BACKBONE_REGISTRY, Backbone, ShapeSpec

from segment_anything import sam_model_registry, SamPredictor

import cv2
import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchvision.datasets import CocoDetection
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from torchvision import transforms


import cv2
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.utils.visualizer import Visualizer
from PIL import Image
import numpy as np

# image_dir = '/home/MBronars/workspace/cs7643-project/images/livecell_train_val_images'
# anno_file = '/home/MBronars/workspace/cs7643-project/train.json'
# register_coco_instances('train_dataset', {}, anno_file, image_dir)

# image_dir = '/home/MBronars/workspace/cs7643-project/images/livecell_train_val_images'
# anno_file = '/home/MBronars/workspace/cs7643-project/valid.json'
# register_coco_instances('valid_dataset', {}, anno_file, image_dir)


# def custom_collate_fn(batch):
#     # Get the images and targets from the batch
#     images = [item[0] for item in batch]
#     targets = [item[1] for item in batch]
    
#     # Convert images to PyTorch tensor
#     images = torch.stack([transforms.ToTensor()(image) for image in images])
    
#     # Combine targets into a single dictionary
#     target_dict = {}
#     for item in targets:
#         target_dict.update(item)
    
#     return images, target_dict

train_dataset = CocoDetection(root='/home/MBronars/workspace/cs7643-project/images/livecell_train_val_images', annFile='/home/MBronars/workspace/cs7643-project/train.json')
val_dataset = CocoDetection(root='/home/MBronars/workspace/cs7643-project/images/livecell_train_val_images', annFile='/home/MBronars/workspace/cs7643-project/train.json')

# # define the dataloaders for training and validation
# train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=2)
# val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=2)

device = "cuda"


class finetuneSAM(nn.Module):
    def __init__(self):
        super(finetuneSAM, self).__init__()
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_C4_1x.yaml"))
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_C4_1x.yaml")
        self.bbox = modeling.build_model(cfg)
        
        # cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/Base-Keypoint-RCNN-FPN.yaml"))
        # cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Keypoints/Base-Keypoint-RCNN-FPN.yaml")
        # self.keypoints = modeling.build_model(cfg)
        
        sam_checkpoint = "/home/MBronars/workspace/segment-anything/segment_data/sam_vit_h_4b8939.pth"
        model_type = "vit_h"
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(device=device)
        
        predictor = SamPredictor(sam)
        predictor.output_mode = "coco_rle"
        self.sam = sam
        
    def forward(self, x):
        temp1 = self.bbox(x)
        # temp2 = self.keypoints(x)
        return self.sam(temp1)

# register the dataset with path to images and annotations
image_dir = '/home/MBronars/workspace/cs7643-project/images/livecell_train_val_images'
anno_file = '/home/MBronars/workspace/cs7643-project/train.json'



    
register_coco_instances('my_dataset', {}, anno_file, image_dir)

# load metadata
metadata = MetadataCatalog.get("my_dataset")

# get the prediction datasetr

dataset_dicts = DatasetCatalog.get("my_dataset")

model = finetuneSAM()

for d in dataset_dicts:
    im = cv2.imread(d["file_name"])
    #image_array = cv2.imdecode(np.frombuffer(im, np.uint8), cv2.IMREAD_COLOR)
    image_tensor = torch.tensor(im).permute(2, 0, 1).unsqueeze(0).float()
    d["image"] = image_tensor.to(device)
    
    #pass the image to the model
model.forward(dataset_dicts)

# from IPython import embed; embed()
# model.forward(dataset_dicts)
# loss = ()
# loss.backward()

# image = cv2.imread("/home/MBronars/workspace/cs7643-project/images/livecell_train_val_images/A172_Phase_A7_1_00d00h00m_1.tif")

# for batch_idx, (data, target) in enumerate(train_loader):   
#     model.forward(data)
#     loss = ()
#     loss.backward()
    
#     model = finetuneSAM()
# for param in model.parameters():
#     print(param)
    

# num_epochs = 10
# for epoch in range(num_epochs):
#     # train for one epoch, printing every 10 iterations
#     train_one_epoch(model, optimizer, train_loader, device, epoch, print_freq=10)
#     # update the learning rate
#     lr_scheduler.step()
#     # evaluate on the test dataset
#     evaluate(model, val_loader, device=device)

# sam_checkpoint = "/home/MBronars/workspace/segment-anything/segment_data/sam_vit_h_4b8939.pth"
# model_type = "vit_h"
# device = "cuda"
# sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
# sam.to(device=device)
# predictor = SamPredictor(sam)
# predictor.output_mode = "coco_rle"

# @BACKBONE_REGISTRY.register()
# class ToyBackbone(Backbone):
#     def forward(self, image):
#         temp = super().forward(image)
#         return predictor(temp)



# cfg.DATALOADER.NUM_WORKERS = 2
#         cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
#         cfg.SOLVER.IMS_PER_BATCH = 2  # This is the real "batch size" commonly known to deep learning people
#         cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
#         cfg.SOLVER.MAX_ITER = 300    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
#         cfg.SOLVER.STEPS = []        # do not decay learning rate
#         cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # The "RoIHead batch size". 128 is faster, and good enough for this toy dataset (default: 512)
#         cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)





# # load configuration
# cfg = get_cfg()
# cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"))
# cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml")
# cfg.MODEL.BACKBONE.NAME = 'ToyBackbone'
# cfg.DATASETS.TRAIN = ("train_dataset",)
# cfg.DATASETS.TEST = ("valid_dataset")
# cfg.DATALOADER.NUM_WORKERS = 2
# cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
# cfg.SOLVER.IMS_PER_BATCH = 2  # This is the real "batch size" commonly known to deep learning people
# cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
# cfg.SOLVER.MAX_ITER = 300    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
# cfg.SOLVER.STEPS = []        # do not decay learning rate
# cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # The "RoIHead batch size". 128 is faster, and good enough for this toy dataset (default: 512)
# cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)

# trainer = DefaultTrainer(cfg) 
# trainer.resume_or_load(resume=False)
# trainer.train()


# # # create model
# # model = modeling.build_model(cfg)
# # model.eval()

# # # define new forward function
# # def new_forward(model, x):
# #     # modify forward pass here
# #     x = model.backbone(x)
# #     x = model.roi_heads(x)
# #     from IPython import embed; embed()
# #     return x

# # # replace original forward function with new_forward
# # model.forward = new_forward