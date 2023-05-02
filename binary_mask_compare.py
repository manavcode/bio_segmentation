
import torch
import numpy as np
import os

from PIL import Image
from torchmetrics import JaccardIndex


scores = []
skipped = []

pred_dir = "/nethome/mlamsey3/Documents/Coursework/cs7643-project/dataset/test/predicted_masks"
actual_dir = "/nethome/mlamsey3/Documents/Coursework/cs7643-project/dataset/test/masks"

n_actual = len([name for name in os.listdir(actual_dir) if os.path.isfile(os.path.join(actual_dir, name))])
n_pred = len([name for name in os.listdir(pred_dir) if os.path.isfile(os.path.join(pred_dir, name))])

for i in range(max([n_actual, n_pred])): #iterate over number of test results
    # actual_path = f'test_latest/images/{i}_actual.png'
    # predicted_path = f'test_latest/images/{i}_predicted.png'

    actual_path = f'{actual_dir}/{i}.jpg'
    predicted_path = f'{pred_dir}/{i}.jpg'

    if not os.path.isfile(actual_path) or not os.path.isfile(predicted_path):
        skipped.append(i)
        continue

    actual = Image.open(actual_path)
    actual = actual.convert("1")
    predicted = Image.open(predicted_path)
    predicted = predicted.convert("1")

    actual_arr = np.array(actual)
    predicted_arr = np.array(predicted)

    actual_tensor = torch.tensor(actual_arr)
    predicted_tensor = torch.tensor(predicted_arr)

    # print(actual_tensor.shape)
    # print(predicted_tensor.shape)

    jaccard = JaccardIndex(task='multiclass', num_classes=2)
    res = jaccard(predicted_tensor, actual_tensor)
    scores.append(res.item())


print("skipped: ", len(skipped))

scores = np.array(scores)
print("mean: ", scores.mean())
print("std: ", scores.std())
print("min: ", scores.min())
print("max: ", scores.max())
