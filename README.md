CS 7643 Deep Learning: Term Project

# Description

A cool project about performing instance segmentation on bright-field microscope images

## Team Members

Manav Agrawal

Matthew Bronars

Vishnu Jaganathan

Matthew Lamsey

# Install

If someone wants to make an installer script, be my guest :)

1. In the root directory, create a folder `/data` (e.g. `cs7643-project/data/`)
2. In the `/data/` directory, create a folder `images/` and a folder for each cell type, e.g. `A172`
3. Download the LIVECell Dataset from https://sartorius-research.github.io/LIVECell/
  - Be sure to download the image `.zip` and the training / test / validation `.json` files
4. Unzip the livecell `images.zip` into the `images/` folder
5. Place `test.json`, `train.json`, and `val.json` into the folder for the corresponding cell

# Notes
The annotations in the LIVECell Dataset appear to follow the [COCO Data Format](https://cocodataset.org/#format-data)
