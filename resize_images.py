# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 15:27:35 2019

@author: rehakomoon
"""

import torchvision
from pathlib import Path
from PIL import Image

input_dir = Path("E:/vrc_rotation/dataset/anotated/")
output_dir = Path("E:/vrc_rotation/dataset/anotated_resized/")

#input_dir = Path("E:/vrc_rotation/dataset/anotated_eval/")
#output_dir = Path("E:/vrc_rotation/dataset/anotated_eval_resized/")

image_size = (480, 480)

output_dir.mkdir(exist_ok=True)

userdir_list = [d.absolute() for d in input_dir.iterdir()]

for userdir in userdir_list:
    print(userdir.stem)
    
    output_user_dir = output_dir / userdir.stem
    output_user_dir.mkdir(exist_ok=True)

    for image_path in  userdir.glob('*.png'):
        filename = image_path.name
        input_path = str(image_path)
        output_path = str(output_user_dir / filename)
        
        image = Image.open(str(image_path))
        max_factor = max([image.size[i]/image_size[i] for i in range(2)])
        image = torchvision.transforms.functional.resize(image,(int(image.size[1]/max_factor), int(image.size[0]/max_factor)))
        
        width, height = image.size
        pad_size = max(width, height)
        image = torchvision.transforms.functional.pad(image, ((pad_size-width) // 2, (pad_size-height) // 2))
        image = torchvision.transforms.functional.resize(image, image_size)
        
        image.save(output_path)
