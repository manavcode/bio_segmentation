
import torch
import numpy as np
import os

from PIL import Image
from torchmetrics import JaccardIndex


scores = []
skipped = []

for i in range(1564): #iterate over number of test results
    actual_path = f'test_latest/images/{i}_actual.png'
    predicted_path = f'test_latest/images/{i}_predicted.png'

    if not os.path.isfile(actual_path) or not os.path.isfile(predicted_path):
        skipped.append(i)
        continue

    actual = Image.open(actual_path)
    predicted = Image.open(predicted_path)

    actual_arr = np.array(actual)
    predicted_arr = np.array(predicted)

    actual_tensor = torch.tensor(actual_arr)
    predicted_tensor = torch.tensor(predicted_arr)

    jaccard = JaccardIndex(task='multiclass', num_classes=2)
    res = jaccard(predicted_tensor, actual_tensor)
    scores.append(res.item())


print("skipped: ", len(skipped))

scores = np.array(scores)
print("mean: ", scores.mean())
print("std: ", scores.std())
print("min: ", scores.min())
print("max: ", scores.max())
