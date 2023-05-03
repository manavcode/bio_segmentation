import argparse
import wandb
import detectron2
from detectron2.data.datasets import register_coco_instances
from detectron2.data import DatasetMapper

# Import necessary modules
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer
from detectron2.model_zoo import model_zoo
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.checkpoint import Checkpointer

import detectron2.utils.comm as comm
from detectron2.engine import hooks

import torch
import os


image_dir = '/home/MBronars/workspace/cs7643-project/images/livecell_train_val_images'
train_file = '/home/MBronars/workspace/cs7643-project/train.json'
val_file = '/home/MBronars/workspace/cs7643-project/val.json'

class TrainLossHook(hooks.HookBase):
    """
    A hook that logs training loss to Weights and Biases after each iteration.
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def after_step(self):
        if self.trainer.iter % self.cfg.LOG_PERIOD == 0:
            # Log training loss to Weights and Biases
            loss = self.trainer.storage.latest()
            wandb.log({"total_loss/train": loss['total_loss'][0]}, step=self.trainer.iter)
            wandb.log({"loss_cls/train": loss['loss_cls'][0]}, step=self.trainer.iter)
            wandb.log({"loss_box_reg/train": loss['loss_box_reg'][0]}, step=self.trainer.iter)
            wandb.log({"loss_mask/train": loss['loss_mask'][0]}, step=self.trainer.iter)
            wandb.log({"loss_rpn_cls/train": loss['loss_rpn_cls'][0]}, step=self.trainer.iter)
            wandb.log({"loss_rpn_loc/train": loss['loss_rpn_loc'][0]}, step=self.trainer.iter)

class ValidationLossHook(hooks.HookBase):
    """
    A hook that calculates validation loss after every epoch.
    Saves model checkpoint if validation loss is the best so far.
    """
    def __init__(self, cfg, val_dataset, check_path):
        super().__init__()
        self.cfg = cfg
        self.dataset = val_dataset
        self.best_loss = float('inf')
        self.checkpoint_path = check_path

    def do_eval(self):
        #Make sure the model is not in training mode
        with torch.no_grad():
            
            avg_loss = None
            total = 0.0
            
            #Need to use the build_detection_train_loader because the default build_detection_test_loader does not return losses
            #Reload the dataset to make sure we are randomly sampling different validation images each time
            loader = detectron2.data.build_detection_train_loader(self.dataset, mapper = DatasetMapper(self.cfg, is_train=True), total_batch_size = self.cfg.SOLVER.IMS_PER_BATCH)
            for i, batch in enumerate(loader):
                total += 1.0
                if i * self.cfg.SOLVER.IMS_PER_BATCH > 1000:
                    break
                val_loss = self.trainer.model(batch)
                if avg_loss is None:
                    avg_loss = {k: 0.0 for k in val_loss.keys()}
                for k in avg_loss.keys():
                    avg_loss[k] += val_loss[k]
            for k in avg_loss.keys():
                avg_loss[k] /= total
            avg_loss['total_loss'] = sum(avg_loss.values())
            
            #Log losses to Weights and Biases
            for k in avg_loss.keys():
                wandb.log({k + '/val': avg_loss[k]}, step = self.trainer.iter)
             
            #Save model checkpoint if validation loss is the best so far
            if avg_loss['total_loss'] < self.best_loss:
                self.best_loss = avg_loss['total_loss']
                checkpointer = Checkpointer(self.trainer.model, save_dir = self.checkpoint_path)
                checkpointer.save(f"model_iter-{self.trainer.iter}_valLoss-{avg_loss['total_loss']}.pth")
                
            #Delete the loader to free up memory
            del loader
            

    def after_step(self):
        if self.trainer.iter % self.cfg.TEST.EVAL_PERIOD != 0:
            return
        # Only perform evaluation on the main process
        if comm.get_rank() > 0:
            return
        self.do_eval()



def train_model(args):

    # Initialize Weights & Biases
    run_name = f"model_{args.model}-batch_{args.batch_size}-lr_{args.lr}-freeze_{args.freeze}"
    wandb.init(project='MaskRCNN-finetuning', name = run_name, config=vars(args))
    
    check_path = f"/home/MBronars/Documents/results/segmentation/{run_name}"
    
    if not os.path.exists(check_path):
        os.makedirs(check_path)
    
    # Register the training and validation datasets
    register_coco_instances("my_training_dataset", {}, train_file, image_dir)
    register_coco_instances("my_validation_dataset", {}, val_file, image_dir)

    # Set metadata for the datasets
    metadata = MetadataCatalog.get("my_training_dataset")
    metadata.set(thing_classes=["cell"], num_classes=1)

    metadata = MetadataCatalog.get("my_validation_dataset")
    metadata.set(thing_classes=["cell"], num_classes=1)
    
    val_dataset = DatasetCatalog.get("my_validation_dataset")
    
    if args.model == 'FPN':
        model = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml"
    elif args.model == 'C4':
        model = "COCO-InstanceSegmentation/mask_rcnn_R_50_C4_1x.yaml"
    elif args.model == 'DC5':
        model = "COCO-InstanceSegmentation/mask_rcnn_R_50_DC5_1x.yaml"

    # Define the configuration
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(model))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model)
    #cfg.merge_from_file(model_zoo.get_config_file(args.model))
    cfg.DATASETS.TRAIN = ("my_training_dataset")
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 4
    cfg.SOLVER.IMS_PER_BATCH = args.batch_size
    cfg.SOLVER.BASE_LR = args.lr
    cfg.SOLVER.MAX_ITER = 10000
    cfg.SOLVER.STEPS = (5000, 7500)
    cfg.SOLVER.GAMMA = 0.1
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    #cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(args.model)
    
    cfg.TEST.EVAL_PERIOD = 100
    cfg.LOG_PERIOD = 10

    # Create the trainer
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    
    for name, param in trainer.model.named_parameters():
        if args.freeze == 'backbone':
            if 'backbone' in name:
                param.requires_grad = False
        elif args.freeze == 'all':
            if 'box_predictor' not in name and 'mask_head' not in name:
                param.requires_grad = False
        elif args.freeze == 'none':
            param.requires_grad = True
        elif args.freeze == 'proposals':
            if 'proposal_generator' in name or 'backbone' in name:
                param.requires_grad = False
    
    trainer.register_hooks([TrainLossHook(cfg), ValidationLossHook(cfg, val_dataset, check_path)])

    # Train the model
    trainer.train()

def main():
    # Set up command line argument parser
    parser = argparse.ArgumentParser(description='Fine-tune a Detectron2 model')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--batch-size', type=int, default=4, help='batch size')
    parser.add_argument('--model', type=str, default='C4', help='model type')
    parser.add_argument('--freeze', type=str, default='default', help='which layers to freeze, possible values: default, backbone, all, none, proposals')
    args = parser.parse_args()

    # Call train_model() function with parsed arguments
    train_model(args)

if __name__ == '__main__':
    main()
