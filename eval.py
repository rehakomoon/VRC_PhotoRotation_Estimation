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
num_image_slice = 8
image_size = (256, 256)

dataset_dir_train = Path("E:/vrc_rotation/dataset/resized/")
dataset_dir_test = Path("E:/vrc_rotation/dataset/resized_validation_aoinu/")

#load_dir = dataset_dir_train
load_dir = dataset_dir_test

save_dir = Path("E:/vrc_rotation/dataset/eval_resized/")
logfile_path = save_dir / "log.txt"
save_dir.mkdir(exist_ok=True)

eval_images_path = [str(p.absolute()) for p in load_dir.glob('*.png')]

for batch_i in range(len(eval_images_path)//batch_size + 1):
    batch_start_idx = batch_i * batch_size
    batch_end_idx = min(batch_start_idx + batch_size, len(eval_images_path))
    if (batch_start_idx == batch_end_idx):
        break
    
    images = []
    
    for idx in range(batch_start_idx, batch_end_idx):
        image = Image.open(eval_images_path[idx])
        if np.floor(min(image.size)/256) >= 2.0:
            factor = int(np.floor(min(image.size)/256))
            image = torchvision.transforms.functional.resize(image, (image.size[0]//factor, image.size[1]//factor))
        
        w, h = image.size
        #print(w, h)
        gap_w = w - image_size[0]
        gap_h = h - image_size[1]
        bias_w = 0 if gap_w > gap_h else gap_w / 2
        bias_h = 0 if gap_h > gap_w else gap_h / 2
        gap_w, gap_h = (gap_w, 0) if gap_w > gap_h else (0, gap_h)
    
        crop_image_list = [
                torchvision.transforms.functional.crop(
                        image,
                        bias_h + i * gap_h / num_image_slice,
                        bias_w + i * gap_w / num_image_slice,
                        256, 256)
                for i in range(num_image_slice)]
        
        rotated_images = [torchvision.transforms.functional.rotate(im, (90 * i)) for im in crop_image_list for i in range(4)]
        images += rotated_images
    
    softmax = nn.Softmax(dim=1)
    
    with torch.no_grad():
        batch_x = torch.cat([torchvision.transforms.ToTensor()(im)[0:3,:,:].view(1, 3, image_size[0], image_size[1]) for im in images], dim=0)
        batch_x = batch_x.cuda()
        
        # [batch*slice*rotation, prob]
        estimated_prob = softmax(model(batch_x))
        
        # [batch*slice, rotation, prob]
        estimated_prob = estimated_prob.reshape(-1, 4, 2)
        #_, estimated = estimated_prob[:, :, 0].max(dim=1)
        log_estimated_prob = estimated_prob.log()
        sum_log_estimated_prob = log_estimated_prob.sum(dim=1, keepdim=True)
        log_likelihood = sum_log_estimated_prob[:,:,1].repeat(1, 4) - log_estimated_prob[:,:,1] + log_estimated_prob[:,:,0]
        #_, estimated = log_likelihood.max(dim=1)
        estimated_prob = softmax(log_likelihood.exp())
        
        # [batch, slice, prob]
        estimated_prob = softmax(log_likelihood.exp()).reshape(-1, num_image_slice, 4)
        a = estimated_prob.prod(dim=1)
        log_estimated_prob = estimated_prob.log()
        log_likelihood = log_estimated_prob.sum(dim=1)
        _, estimated = log_likelihood.max(dim=1)
        #estimated_prob_all = softmax(log_likelihood.exp())
        estimated_prob_all = softmax(log_likelihood)
        
        estimated = estimated.cpu().detach()
        estimated_prob_all = estimated_prob_all.cpu().detach()
    
    for i, idx in enumerate(range(batch_start_idx, batch_end_idx)):
        estimated_rotation = int(estimated[i])
        estimated_prob = estimated_prob_all[i,:].numpy()
        image = cv2.imread(eval_images_path[idx])
        if (estimated_rotation > 0):
            image = cv2.rotate(image, 3-estimated_rotation)
        save_path = save_dir / Path(eval_images_path[idx]).name
        
        # for debugging
        #cv2.imwrite(str(save_path), image)
        if (estimated_rotation > 0):
            cv2.imwrite(str(save_path), image)
        output_string = f"{Path(eval_images_path[idx]).name}: {90*estimated_rotation}, [{', '.join([f'{p:.3f}' for p in estimated_prob])}]"
        print(output_string)
        with open(logfile_path, "a") as fout:
            fout.write(output_string+"\n")
