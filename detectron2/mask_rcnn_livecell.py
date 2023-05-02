import torch, detectron2
from torchsummary import summary
TORCH_VERSION = ".".join(torch.__version__.split(".")[:2])
CUDA_VERSION = torch.__version__.split("+")[-1]

# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random
# from google.colab.patches import cv2_imshow

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.modeling import build_model
from detectron2.evaluation.coco_evaluation import instances_to_coco_json
from pycocotools.coco import COCO

# matt
import matplotlib.pyplot as plt
from torchmetrics.classification import JaccardIndex

# set up livecell
from detectron2.data.datasets import register_coco_instances
# train_json = "/nethome/mlamsey3/Documents/Coursework/cs7643-project/data/A172/train.json"
# val_json = "/nethome/mlamsey3/Documents/Coursework/cs7643-project/data/A172/val.json"
# test_json = "/nethome/mlamsey3/Documents/Coursework/cs7643-project/data/A172/test.json"
train_json = "/nethome/mlamsey3/Documents/Coursework/cs7643-project/data/livecell_coco_train.json"
val_json = "/nethome/mlamsey3/Documents/Coursework/cs7643-project/data/livecell_coco_val.json"
test_json = "/nethome/mlamsey3/Documents/Coursework/cs7643-project/data/livecell_coco_test.json"
train_val_imgs = "/nethome/mlamsey3/Documents/Coursework/cs7643-project/data/images/livecell_train_val_images"
test_imgs = "/nethome/mlamsey3/Documents/Coursework/cs7643-project/data/images/livecell_test_images"

register_coco_instances("livecell_train", {}, train_json, train_val_imgs)
register_coco_instances("livecell_val", {}, val_json, train_val_imgs)
register_coco_instances("livecell_test", {}, test_json, test_imgs)

def cv2_imshow(im, bool_matplotlib=True):
    if bool_matplotlib:
        plt.imshow(im)
        plt.show()
    else:
        cv2.imshow("img", im)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

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

def train(cfg):
    # set up trainer
    from detectron2.engine import DefaultTrainer

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg) 
    trainer.resume_or_load(resume=False)
    trainer.train()

def visualize(cfg):
    # output
    print(cfg.OUTPUT_DIR)
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5   # set a custom testing threshold
    predictor = DefaultPredictor(cfg)
    img1 = cv2.imread("/nethome/mlamsey3/Documents/Coursework/cs7643-project/dataset/train/images/6.jpg")
    img2 = cv2.imread("/nethome/mlamsey3/lamsey_2.jpg")
    # img = cv2.imread("./input.jpg")
    # img = cv2.imread("/nethome/mlamsey3/Documents/Coursework/cs7643-project/detectron2/test_img/test.jpg")

    outputs1 = predictor(img1)
    outputs2 = predictor(img2)

    img = cv2.imread("/nethome/mlamsey3/Documents/Coursework/cs7643-project/dataset/train/masks/6.jpg")
    v = Visualizer(img[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    out1 = v.draw_instance_predictions(outputs1["instances"].to("cpu"))
    out1_img = out1.get_image()[:, :, ::-1]

    img = cv2.imread("/nethome/mlamsey3/Documents/Coursework/cs7643-project/dataset/train/images/6.jpg")
    v = Visualizer(img[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    out2 = v.draw_instance_predictions(outputs1["instances"].to("cpu"))
    out2_img = out2.get_image()[:, :, ::-1]

    vertical_grey_line = np.ones((out1_img.shape[0], 5, 3)) * 128
    vertical_grey_line = vertical_grey_line.astype(np.uint8)

    # out = cv2.hconcat([out1_img, vertical_grey_line, out2_img])
    out = cv2.vconcat([out2_img, out1_img])

    cv2_imshow(out, bool_matplotlib=False)

def performance(cfg):
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.0001   # set a custom testing threshold
    predictor = DefaultPredictor(cfg)
    img = cv2.imread("/nethome/mlamsey3/Documents/Coursework/cs7643-project/dataset/train/images/1.jpg")

    outputs = predictor(img)
    for key, _ in outputs["instances"].items():
        print(key)

    #####
    jaccard = JaccardIndex(task="binary", threshold=0.5)
    # iou = jaccard(pred, target)

def predict_train(predictor, filename="new_coco.json"):
    # init
    train_json = "/nethome/mlamsey3/Documents/Coursework/cs7643-project/data/A172/train.json"
    data_dir = "/nethome/mlamsey3/Documents/Coursework/cs7643-project/data/images/livecell_train_val_images/"

    # coco
    old_coco = COCO(train_json)
    new_coco = {}
    new_coco['info'] = old_coco.dataset['info']
    new_coco['licenses'] = old_coco.dataset['licenses']
    new_coco['categories'] = old_coco.dataset['categories']
    new_coco['annotations'] = []
    new_coco['images'] = []

    with open(train_json) as f:
        train = json.load(f)
        annotation_i = 0
        for image in train["images"]:
        # for image in [train["images"][0]]:
            id = image["id"]
            filename = image["file_name"]
            img = cv2.imread(data_dir + filename)
            outputs = predictor(img)
            instances = outputs["instances"].to("cpu")
            coco_json = instances_to_coco_json(instances, id)
            # print(coco_json[0].keys())
            new_coco["images"].append(image)
            for instance in coco_json:
                # print(instance.keys())
                area = instance["bbox"][2] * instance["bbox"][3]
                # print(instance["bbox"])
                new_ann = {
                    # 'image_id': instance['image_id'],
                    'image_id': id,
                    'id': annotation_i,
                    'category_id': 1,
                    'segmentation': instance['segmentation'],
                    'area': area,
                    'bbox': instance['bbox'],
                    'iscrowd': 0,
                    'score': instance['score']
                }
                new_coco['annotations'].append(new_ann)
                annotation_i += 1
        
        # dump json to file
        with open(filename, 'w') as outfile:
            json.dump(new_coco, outfile)
            

class Freezer:
    def __init__(self):
        """
        Class to freeze layers of a model
        """

        self.freezes = [self.freeze1, self.freeze2, self.freeze3, self.freeze4]

    @staticmethod
    def freeze1(model):
        """
        Freeze all layers except for the last one (predictor)
        """
        for name, param in model.named_parameters():
            if "predictor" not in name:
                param.requires_grad = False
    
    @staticmethod
    def freeze2(model):
        """
        Freeze all layers except for the deconvolution and predictor layers
        """
        for name, param in model.named_parameters():
            if "predictor" not in name and "deconv" not in name:
                param.requires_grad = False

    @staticmethod
    def freeze3(model):
        """
        Freeze all layers except for one mask upscaling, deconv, and the predictor layers
        """
        for name, param in model.named_parameters():
            if "predictor" not in name and "mask_fcn4" not in name and "deconv" not in name:
                param.requires_grad = False

    @staticmethod
    def freeze4(model):
        """
        Freeze all layers except for all mask upscaling, deconv and the predictor layers
        """
        for name, param in model.named_parameters():
            if "predictor" not in name and "mask_fcn" not in name and "deconv" not in name:
                param.requires_grad = False


if __name__ == '__main__':
    cfg = get_train_cfg()

    # freeze_ids = [0, 1, 2, 3]
    # freezer = Freezer()
    # for i in freeze_ids:
    #     print("TRAINING MODEL WITH FREEZE " + str(i) + "...")
    #     cfg.OUTPUT_DIR = "/nethome/mlamsey3/Documents/Coursework/cs7643-project/detectron2/output_test_freeze_" + str(i)
    #     model = build_model(cfg)
    #     freezer.freezes[i](model)
    #     train(cfg)

    cfg.OUTPUT_DIR = "/nethome/mlamsey3/Documents/Coursework/cs7643-project/detectron2/output_no_freeze"
    # model = build_model(cfg)
    # print(model)
    # for name, param in model.named_parameters():
    #     print(name)
    # exit()

    # # freeze all layers except for the last one (predictor)
    # for name, param in model.named_parameters():
    #     if "predictor" not in name:
    #         param.requires_grad = False

    # train
    train(cfg)

        # predictor = DefaultPredictor(cfg)
        # filename = "new_coco_" + str(i) + ".json"
        # predict_train(predictor, filename=filename)

    visualize(cfg)

    