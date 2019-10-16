# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 15:27:35 2019

@author: rehakomoon
"""

import cv2
import numpy as np
from pathlib import Path

input_dir = Path("E:/vrc_rotation/dataset/validation_aoinu/")
output_dir = Path("E:/vrc_rotation/dataset/resized_valdation_aoinu/")

output_dir.mkdir(exist_ok=True)

for i, image_path in enumerate(input_dir.iterdir()):
    print(i)
    img = cv2.imread(str(image_path))
    im_height, im_width = img.shape[0:2]
    output_filename = output_dir / f"{i:04}.png"
    output_im_size = (int(im_width/4), int(im_height/4))
    resized_img = cv2.resize(img, output_im_size)
    cv2.imwrite(str(output_filename), resized_img)
