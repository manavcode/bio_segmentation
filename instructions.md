# STEPS TO SETUP
1. install coco python module, assuming tensorflow etc is there
2. download iamges zip file from https://sartorius-research.github.io/LIVECell/ where it says 'this link'
3. follow the curl instructions to get train/test/val json files
4. run the make_dataset.py script
5. copy either maskrcnn or transformer folder and modify to implement your chosen architecture