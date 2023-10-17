from PIL import Image
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
import pandas as pd
import shutil

os.chdir('/storage/yskim/lesion/train_masks')
train_df = pd.read_csv('/storage/yskim/lesion/train.csv')
maskpath = os.listdir()
maskpath.sort()

for path in maskpath:
    img = Image.open('/storage/yskim/lesion/ori_train_image/' + path).convert('L')
    #
    # Save
    img.save(f'/storage/yskim/lesion/ori_train_image_gray/{path}')

print('finish')
