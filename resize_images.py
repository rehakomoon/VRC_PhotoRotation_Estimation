# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 15:27:35 2019

@author: rehakomoon
"""

import cv2
import numpy as np
from pathlib import Path
import shutil

userlist = ["hakomoon"]

output_dir = Path("E:/vrc_rotation/dataset/collection/")
output_dir.mkdir(exist_ok=True)

image_idx = 0

for username in userlist:
    input_dir = Path("E:/vrc_rotation/dataset/anotated/" + username + "/")
    for image_path in input_dir.iterdir():
        if image_idx%100 == 0:
            print(image_idx)
        #img = cv2.imread(str(image_path))
        #im_height, im_width = img.shape[0:2]
        shutil.copy(str(image_path), str(output_dir / f"{image_idx:06}.png"))
        image_idx += 1
        #output_im_size = (int(im_width/4), int(im_height/4))
        #resized_img = cv2.resize(img, output_im_size)
        #cv2.imwrite(str(output_filename), resized_img)
