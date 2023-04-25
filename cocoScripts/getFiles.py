#Use this script if you need to download any of the files from the LiveCell dataset
import requests

#Comment out any section that you don't need to download

#Download the images
url = "http://livecell-dataset.s3.eu-central-1.amazonaws.com/LIVECell_dataset_2021/images.zip"
response = requests.get(url)

with open("liveCellImages.zip", "wb") as f:
    f.write(response.content)

#Download the train annotations
url = "http://livecell-dataset.s3.eu-central-1.amazonaws.com/LIVECell_dataset_2021/annotations/LIVECell/livecell_coco_train.json"
response = requests.get(url)

with open("trainImgs.json", "wb") as f:
    f.write(response.content)

#Download the val annotations    
url = "http://livecell-dataset.s3.eu-central-1.amazonaws.com/LIVECell_dataset_2021/annotations/LIVECell/livecell_coco_val.json"
response = requests.get(url)

with open("valImgs.json", "wb") as f:
    f.write(response.content)

#Download the test annotations
url = "http://livecell-dataset.s3.eu-central-1.amazonaws.com/LIVECell_dataset_2021/annotations/LIVECell/livecell_coco_test.json"
response = requests.get(url)

with open("testImgs.json", "wb") as f:
    f.write(response.content)
