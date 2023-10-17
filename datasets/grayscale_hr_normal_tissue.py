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
    # label = Image.open('/storage/yskim/lesion/gt_train_image/' + path)
    # img = Image.open('/storage/yskim/lesion/ori_train_image/' + path)
    label = np.array(Image.open('/storage/yskim/lesion/gt_train_image_orig/' + path))
    img = cv2.imread('/storage/yskim/lesion/ori_train_image/' + path)
    #
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    lower_red = np.array([140, 10, 10])
    upper_red = np.array([220, 255, 255])
    red_mask = cv2.inRange(hsv, lower_red, upper_red)
    onlyRED = cv2.bitwise_and(img, img, mask=red_mask)
    # Normal Tissue, class_id = 1
    gt_label = np.zeros([img.shape[0], img.shape[1]])
    gt_label[(onlyRED.sum(axis=2) != 0)] = 1
    # ===== Debugging =====
    # a = (np.concatenate([np.expand_dims(gt_label, axis=2), np.expand_dims(gt_label, axis=2), np.expand_dims(gt_label, axis=2)], axis=2) * 255).astype(np.uint8)
    # plt.imshow(np.concatenate([img, a], axis=1))
    # plt.show()
    # =========================
    #
    # Invasive cancer, class_id = 1 -> 2
    # Tumor, class_id = 2 -> 3
    # DCIS non comedo, class_id = 3 -> 4
    # DCIS comedo, class_id = 4 -> 5
    # LCIS, class_id 5 -> 6
    gt_label += label
    # Save
    img_pil = Image.fromarray(img.astype(np.uint8), 'RGB')
    gt_label_pil = Image.fromarray(gt_label.astype(np.uint8), 'L')
    # maskedimg_ = Image.fromarray(np.squeeze((gtimg/gtimg.max()*255).astype(np.uint8), axis=2), 'L')
    # maskimg_orig = Image.fromarray(maskimg)

    gt_label_pil.save(f'/storage/yskim/lesion/gt_train_image/{path}')
    # img_pil.save(f'/storage/yskim/lesion/ori_train_image_hr/{path}')

print('finish')
