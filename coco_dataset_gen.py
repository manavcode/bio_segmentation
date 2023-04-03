from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import random
from PIL import Image


def getImage(imageObj, img_folder, input_image_size):

    train_img = io.imread(img_folder + '/' + imageObj['file_name'])

    # resizing image to 256 x 256 (was needed for pix2pix)
    train_img = cv2.resize(train_img, (346, 256))
    train_img = train_img[:,:256]

    # its a grayscale image, but turned into rgb format for pix2pix
    stacked_img = np.stack((train_img,)*3, axis=-1)

    return stacked_img


def getBinaryMask(imageObj, coco, input_image_size):
    annIds = coco.getAnnIds(imageObj['id'])
    anns = coco.loadAnns(annIds)
    train_mask = np.zeros((256, 256))
    for a in range(len(anns)):
        new_mask = coco.annToMask(anns[a])

        # again resizing specific to pix2pix
        new_mask = cv2.resize(new_mask, (346,256))
        new_mask = new_mask[:,:256]

        # Threshold because resizing may cause extraneous values
        new_mask[new_mask >= 0.5] = 255
        new_mask[new_mask < 0.5] = 0

        train_mask = np.maximum(new_mask, train_mask)

    # make b/w image to rgb for pix2pix
    stacked_mask = np.stack((train_mask,)*3, axis=-1)

    return stacked_mask


def dataGeneratorCoco():

    annFile = 'livecell_coco_test.json'
    # annFile = 'livecell_coco_train.json'
    # annFile = 'livecell_coco_val.json'

    coco = COCO(annFile)

    imgIds = coco.getImgIds()
    images = coco.loadImgs(imgIds)

    input_image_size=(520, 704)
    batch_size = 4

    img_folder = 'images/livecell_test_images'
    # img_folder = 'images/livecell_train_val_images'
    dataset_size = len(images)

    random.shuffle(images)

    for i in range(dataset_size):

        imageObj = images[i]

        # get image
        train_img = getImage(imageObj, img_folder, input_image_size)

        # get mask
        train_mask = getBinaryMask(imageObj, coco, input_image_size)

        # make sure arrays are right type
        img = train_img.astype('uint8')
        mask = train_mask.astype('uint8')

        # side by side images for pix2pix
        # res = np.concatenate((img, mask), axis=1)
        # img = Image.fromarray(res, "RGB")
        # img.save(str(i) + ".jpg")

        # can save just the mask to a seperate 'test' folder or use the following if
        # you want to yield from a generator:
        # yield img, mask
