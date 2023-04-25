import os
import json
import cv2
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.utils.visualizer import Visualizer
from PIL import Image
import numpy as np

# register the dataset with path to images and annotations
image_dir = '/path/to/livecell_test_images'
anno_file = '/path/to/segResult.json'
register_coco_instances('my_dataset', {}, anno_file, image_dir)

# load metadata
metadata = MetadataCatalog.get("my_dataset")

# get the prediction dataset
dataset_dicts = DatasetCatalog.get("my_dataset")

for d in dataset_dicts:
    # read the image
    img = np.array(Image.open(d["file_name"]))

    # create a visualizer
    visualizer = Visualizer(img, metadata=metadata, scale=1.0)

    # draw the annotations on the image
    vis = visualizer.draw_dataset_dict(d)

    # convert the visualizer output to an RGB image
    vis = vis.get_image()

    # save the image
    # currently overwrites the image, but you can change the name to save multiple images in a folder
    cv2.imwrite("test.png", vis[:, :, ::-1])
