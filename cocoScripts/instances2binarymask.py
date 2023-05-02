import os
import cv2
import numpy as np
import torch
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import Visualizer
from detectron2 import model_zoo

def get_train_cfg():
    n_train_val_livecell = 3725
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("livecell_train",)
    cfg.DATASETS.TEST = ("livecell_val",)  # note! just added this
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
    # cfg.SOLVER.IMS_PER_BATCH = 2  # This is the real "batch size" commonly known to deep learning people
    cfg.SOLVER.IMS_PER_BATCH = 4  # This is the real "batch size" commonly known to deep learning people
    cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
    cfg.SOLVER.MAX_ITER = n_train_val_livecell * 2    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
    cfg.SOLVER.STEPS = []        # do not decay learning rate
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # The "RoIHead batch size". 128 is faster, and good enough for this toy dataset (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
    # NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.
    cfg.TEST.DETECTIONS_PER_IMAGE = 2000
    return cfg

def get_instance_predictions(image_path, model_weights):
    cfg = get_train_cfg()
    # cfg = get_cfg()
    # cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = model_weights
    # cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.05
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    # cfg.MODEL.DEVICE = "cpu"

    predictor = DefaultPredictor(cfg)
    image = cv2.imread(image_path)
    outputs = predictor(image)
    # print(outputs)

    v = Visualizer(image[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    # cv2.imwrite("/nethome/mlamsey3/Documents/Coursework/cs7643-project/cocoScripts/predictions.png", out.get_image()[:, :, ::-1])
    print(os.path.basename(image_path))
    cv2.imwrite("/nethome/mlamsey3/Documents/Coursework/cs7643-project/dataset/test/predicted_instances/{}".format(os.path.basename(image_path)), out.get_image()[:, :, ::-1])
    # print(len(outputs["instances"].pred_masks))

    return image, outputs["instances"]

def create_binary_mask(image, instances, output_path):
    print(len(instances.pred_masks))
    height, width, _ = image.shape
    binary_mask = np.zeros((height, width), dtype=np.uint8)

    for mask in instances.pred_masks:
        binary_mask += mask.cpu().numpy().astype(np.uint8)

    binary_mask = np.clip(binary_mask, 0, 1) * 255
    return binary_mask
    # cv2.imwrite(output_path, binary_mask)

def create_binary_mask_from_image(image_path, model_weights, output_path):
    image, instances = get_instance_predictions(image_path, model_weights)
    return create_binary_mask(image, instances, output_path)

def create_binary_masks_dir(input_dir, model_weights, output_dir):
    for filename in os.listdir(input_dir):
        image_path = os.path.join(input_dir, filename)
        get_instance_predictions(image_path, model_weights)
        output_path = os.path.join(output_dir, filename)
        # mask = create_binary_mask_from_image(image_path, model_weights, output_path)

        # cv2.imwrite(output_path, mask)

if __name__ == "__main__":
    # image_path = "/nethome/mlamsey3/Documents/Coursework/cs7643-project/data/images/livecell_test_images/A172_Phase_C7_1_00d00h00m_2.tif"
    # image_path = "/nethome/mlamsey3/Documents/Coursework/cs7643-project/dataset/test/images/1.jpg"
    # model_weights = "/nethome/mlamsey3/Documents/Coursework/cs7643-project/detectron2/output/model_final.pth"
    model_weights = "/nethome/mlamsey3/Documents/Coursework/cs7643-project/detectron2/output_no_freeze/model_final.pth"
    # output_path = "/nethome/mlamsey3/Documents/Coursework/cs7643-project/cocoScripts/binary_mask.png"

    # image, instances = get_instance_predictions(image_path, model_weights)
    # create_binary_mask(image, instances, output_path)

    prediction_output_dir = "/nethome/mlamsey3/Documents/Coursework/cs7643-project/dataset/test/predicted"
    input_dir = "/nethome/mlamsey3/Documents/Coursework/cs7643-project/dataset/test/images"
    create_binary_masks_dir(input_dir, model_weights, prediction_output_dir)

