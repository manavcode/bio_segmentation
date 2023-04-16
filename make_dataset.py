from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import random
from PIL import Image


def getImage(imageObj, img_folder):
    train_img = io.imread(img_folder + '/' + imageObj['file_name'])
    stacked_img = np.stack((train_img,)*3, axis=-1)
    return stacked_img


def getBinaryMask(imageObj, coco, input_image_size):
    annIds = coco.getAnnIds(imageObj['id'])
    anns = coco.loadAnns(annIds)
    train_mask = np.zeros(input_image_size)
    for a in range(len(anns)):
        new_mask = coco.annToMask(anns[a])*255
        train_mask = np.maximum(new_mask, train_mask)

    stacked_mask = np.stack((train_mask,)*3, axis=-1)
    return stacked_mask


def dataGeneratorCoco(dataset_type):

    if dataset_type == 'train':
        annFile = 'coco_json/livecell_coco_train.json'
        img_folder = 'images/livecell_train_val_images'
        path = 'dataset/train/'
    elif dataset_type == 'test':
        annFile = 'coco_json/livecell_coco_test.json'
        img_folder = 'images/livecell_test_images'
        path = 'dataset/test/'
    elif dataset_type == 'val':
        annFile = 'coco_json/livecell_coco_val.json'
        img_folder = 'images/livecell_train_val_images'
        path = 'dataset/test/'


    coco = COCO(annFile)

    imgIds = coco.getImgIds()
    images = coco.loadImgs(imgIds)

    input_image_size=(520, 704)
    
    dataset_size = len(images)

    random.shuffle(images)
    print("Processing: ", dataset_size)
    for i in range(dataset_size):

        imageObj = images[i]

        # get img
        train_img = getImage(imageObj, img_folder)

        # get mask
        train_mask = getBinaryMask(imageObj, coco, input_image_size)

        img = train_img.astype('uint8')
        img = Image.fromarray(img, 'RGB')
        img.save(path + 'images/' + str(i) + ".jpg")

        mask = train_mask.astype('uint8')
        mask = Image.fromarray(mask, "RGB")
        mask.save(path + 'masks/' + str(i) + ".jpg")

        if i % 10 == 0:
            print(str(i) + " completed")


dataGeneratorCoco('train')
dataGeneratorCoco('test')
dataGeneratorCoco('val')



