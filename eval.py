# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 19:28:07 2019

@author: rehakomoon
"""

from pathlib import Path
import torch
import torchvision
from PIL import Image
import numpy as np

import torch.nn as nn

from model import Model

import cv2

model_dir = Path("E:/vrc_rotation/log/")

model_epoch_list = [int(str(s)[-10:-4]) for s in model_dir.glob("model_*.pth")]
latest_model_path = model_dir / f"model_{max(model_epoch_list):06}.pth"

print(f"load {latest_model_path}...")
model_params = torch.load(str(latest_model_path))

model = Model()
model.load_state_dict(model_params)
model = model.cuda()
model.eval()

batch_size = 16
image_size = (480, 480)

dataset_dir_train = Path("E:/vrc_rotation/dataset/anotated/")
dataset_dir_test = Path("E:/vrc_rotation/dataset/anotated_eval/")
dataset_dir_eval = Path("E:/vrc_rotation/dataset/eval/")

#dataset_dir = dataset_dir_train
#dataset_dir = dataset_dir_test
dataset_dir = dataset_dir_eval
#dataset_dir = Path("E:/vrc_gan/dataset_org_mbrn/tmp/")

for user_dir in dataset_dir.iterdir():
    username = user_dir.stem
    
    save_dir = Path(f"E:/vrc_rotation/dataset/eval_{username}/")
    t_save_dir = Path(f"E:/vrc_rotation/dataset/eval_{username}_t/")
    logfile_path = save_dir / "log.txt"
    save_dir.mkdir(exist_ok=True)
    t_save_dir.mkdir(exist_ok=True)
    
    eval_images_path = [str(p.absolute()) for p in user_dir.glob('*.png')]
    num_images = len(eval_images_path)
    num_error = 0
    
    for batch_i in range(len(eval_images_path)//batch_size + 1):
        batch_start_idx = batch_i * batch_size
        batch_end_idx = min(batch_start_idx + batch_size, len(eval_images_path))
        if (batch_start_idx == batch_end_idx):
            break
        
        images = []
        
        for idx in range(batch_start_idx, batch_end_idx):
            image = Image.open(eval_images_path[idx])
            width, height = image.size
            pad_size = max(width, height)
            image = torchvision.transforms.functional.pad(image, ((pad_size-width) // 2, (pad_size-height) // 2))
            image = torchvision.transforms.functional.resize(image, image_size)
            
            rotated_images = [torchvision.transforms.functional.rotate(image, (90 * i)) for i in range(4)]
            images += rotated_images
            
            image = torchvision.transforms.functional.hflip(image)
            rotated_images = [torchvision.transforms.functional.rotate(image, (90 * i)) for i in range(4)]
            images += rotated_images
        
        softmax = nn.Softmax(dim=1)
        
        with torch.no_grad():
            batch_x = torch.cat([torchvision.transforms.ToTensor()(im)[0:3,:,:].view(1, 3, image_size[0], image_size[1]) for im in images], dim=0)
            batch_x = batch_x.cuda()
            
            # [batch*rotation, prob]
            estimated_prob = softmax(model(batch_x))
            
            # [batch, flip, rotation, prob]
            estimated_prob = estimated_prob.reshape(-1, 2, 4, 2)
            #_, estimated = estimated_prob[:, :, 0].max(dim=1)
            log_estimated_prob = estimated_prob.log()
            
            # [batch, flip, 1, prob]
            sum_log_estimated_prob = log_estimated_prob.sum(dim=2, keepdim=True)
            
            # [batch, flip, rotation, prob] -> [batch, flip, rotation]
            log_likelihood = sum_log_estimated_prob[:,:,:,1].repeat(1, 1, 4) - log_estimated_prob[:,:,:,1] + log_estimated_prob[:,:,:,0]

            # [batch, flip, rotation] -> [batch, rotation]
            log_likelihood = log_likelihood.sum(dim=1)
            
            # [batch, rotation]
            _, estimated = log_likelihood.max(dim=1)
            #estimated_prob = softmax(log_likelihood.exp())
            estimated_prob = softmax(log_likelihood)
            
            estimated = estimated.cpu().detach()
            estimated_prob = estimated_prob.cpu().detach()
        
        for i, idx in enumerate(range(batch_start_idx, batch_end_idx)):
            estimated_rotation = int(estimated[i])
            estimated_prob_i = estimated_prob[i,:].numpy()

            #save_flag = True
            save_flag = False
            
            if (estimated_rotation > 0):
                save_flag = True
                num_error += 1
            
            if (save_flag):
                image = cv2.imread(eval_images_path[idx])
                if (estimated_rotation > 0):
                    image = cv2.rotate(image, 3-estimated_rotation)
                save_path = save_dir / Path(eval_images_path[idx]).name
                cv2.imwrite(str(save_path), image)
            
            if True:
                if (estimated_prob_i[0] < 0.9 and estimated_rotation == 0):
                    image = cv2.imread(eval_images_path[idx])
                    save_path = t_save_dir / Path(eval_images_path[idx]).name
                    cv2.imwrite(str(save_path), image)

            output_string = f"{Path(eval_images_path[idx]).name}: {90*estimated_rotation}, [{', '.join([f'{p:.3f}' for p in estimated_prob_i])}]"
            print(output_string)
            with open(logfile_path, "a") as fout:
                fout.write(output_string+"\n")
        
    output_string = f"{num_images} images evaluated. {num_error} images rotated. {num_error/num_images}, {1.0-num_error/num_images}"
    print(output_string)
    with open(logfile_path, "a") as fout:
        fout.write(output_string+"\n")
