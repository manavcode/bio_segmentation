import os
import json
import numpy as np

import matplotlib.pyplot as plt

data_path = os.path.expanduser("~") + "/Documents/Coursework/cs7643-project/data/"
a172_train_annotations_path = data_path + "A172/train.json"

with open(a172_train_annotations_path, 'r') as f:
    a172_train = json.load(f)

images = a172_train["images"]
image_names = [img["file_name"] for img in images]
sorted_indices = np.argsort(image_names)

f = plt.figure()

for i in range(30):
    _i = sorted_indices[i]
    img_info = images[_i]
    file_name = img_info["file_name"]
    file_path = data_path + "images/livecell_train_val_images/" + file_name
    # print(file_path)
    img = plt.imread(file_path)
    plt.imshow(img)
    plt.title(file_name)
    plt.show()

# p = f.subplots(2, 2)
# j = [3, 4, 5, 6]
# for i in range(len(j)):
#     _i = sorted_indices[j[i]]
#     img_info = images[_i]
#     file_name = img_info["file_name"]
#     file_path = data_path + "images/livecell_train_val_images/" + file_name
#     img = plt.imread(file_path)
#     plot = p[i // 2, i % 2]
#     plot.imshow(img)
#     plot.set_title(file_name)

# plt.show()
