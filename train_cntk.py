# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 20:47:27 2019

@author: rehakomoon
"""

from pathlib import Path
import random
import itertools
import numpy as np
from PIL import Image, ImageOps
import cntk as C
from tqdm import tqdm


dataset_train_dir = Path("C:/dataset/anotated_resized/")
dataset_test_dir = Path("E:/vrc_rotation/dataset/anotated_eval_resized/")
log_dir = Path("E:/vrc_rotation/log_cntk/")
logfile_path = Path("E:/vrc_rotation/log_cntk/log.txt")

log_dir.mkdir(exist_ok=True)

batch_size = 32
test_batch_size = batch_size // 8
num_epoch = 10000
initial_epoch = 0
learning_rate = 0.001

image_size = 480

def get_image_list(dataset_dir):
    image_path_list = []
    dataset_dir = Path(dataset_dir)
    for user_dir in dataset_dir.iterdir():
        image_path_list += [str(p.absolute())  for p in user_dir.glob('*.png')]
    return image_path_list

train_image_path_list = get_image_list(dataset_train_dir)
test_image_path_list = get_image_list(dataset_test_dir)

x_pl = C.ops.input_variable((3, image_size, image_size), np.float32)
y_pl = C.ops.input_variable((2), np.float32)

def CNN(x):
    with C.layers.default_options(init=C.initializer.glorot_uniform()):
        x = C.layers.Convolution2D(filter_shape=(5,5), num_filters=16, activation=None)(x)
        x = C.layers.BatchNormalization(map_rank=1)(x)
        x = C.relu(x)
        x = C.layers.MaxPooling(filter_shape=(2,2), strides=(2,2))(x)
        x = C.layers.Convolution2D(filter_shape=(5,5), num_filters=16, activation=None)(x)
        x = C.layers.BatchNormalization(map_rank=1)(x)
        x = C.relu(x)
        x = C.layers.MaxPooling(filter_shape=(2,2), strides=(2,2))(x)
        x = C.layers.Convolution2D(filter_shape=(5,5), num_filters=64, activation=None)(x)
        x = C.layers.BatchNormalization(map_rank=1)(x)
        x = C.relu(x)
        x = C.layers.MaxPooling(filter_shape=(2,2), strides=(2,2))(x)
        x = C.layers.Dropout(0.3)(x)
        x = C.layers.Convolution2D(filter_shape=(5,5), num_filters=64, activation=None)(x)
        x = C.layers.BatchNormalization(map_rank=1)(x)
        x = C.relu(x)
        x = C.layers.MaxPooling(filter_shape=(2,2), strides=(2,2))(x)
        x = C.layers.Dropout(0.3)(x)
        x = C.layers.Convolution2D(filter_shape=(5,5), num_filters=256, activation=None)(x)
        x = C.layers.BatchNormalization(map_rank=1)(x)
        x = C.relu(x)
        x = C.layers.MaxPooling(filter_shape=(2,2), strides=(2,2))(x)
        x = C.layers.Dropout(0.3)(x)
        x = C.layers.Convolution2D(filter_shape=(5,5), num_filters=256, activation=None)(x)
        x = C.layers.BatchNormalization(map_rank=1)(x)
        x = C.relu(x)
        x = C.layers.MaxPooling(filter_shape=(2,2), strides=(2,2))(x)
        x = C.layers.Dropout(0.3)(x)
        x = C.layers.MaxPooling(filter_shape=(3,3), strides=(1,1))(x)
        x = C.layers.Dense(256, activation=None)(x)
        x = C.relu(x)
        x = C.layers.Dropout(0.3)(x)
        x = C.layers.Dense(256, activation=None)(x)
        x = C.relu(x)
        x = C.layers.Dropout(0.3)(x)
        x = C.layers.Dense(2, activation=None)(x)
    return x

model = CNN(x_pl)

lr_schedule = C.learners.learning_rate_schedule(learning_rate, unit=C.UnitType.sample)
optimizer = C.learners.sgd(model.parameters, lr=lr_schedule)

loss = C.losses.cross_entropy_with_softmax(model, y_pl)
acc = C.metrics.classification_error(model, y_pl)

trainer = C.Trainer(model, (loss, acc), optimizer)

model_epoch_list = [int(str(s)[-10:-4]) for s in log_dir.glob("model_*.dat")]
if (len(model_epoch_list) > 0):
    latest_model_path = log_dir / f"model_{max(model_epoch_list):06}.dat"
    print(f"load {latest_model_path}...")
    state = trainer.restore_from_checkpoint(str(latest_model_path))
    initial_epoch = max(model_epoch_list) + 1

for epoch in range(initial_epoch, num_epoch):
    image_path_list = train_image_path_list
    random.shuffle(image_path_list)
    
    sum_loss = 0.0
    sum_acc = 0.0
    sum_seen = 0.0
    
    my_bar = tqdm(range(0, len(image_path_list), batch_size), leave=False)
    for i in my_bar:
        batch_image_path_list = image_path_list[i:i+batch_size]
        this_batch_size = len(batch_image_path_list)

        rotate_angle = np.random.randint(0, 6, this_batch_size)
        rotate_angle[rotate_angle > 3] = 0
        flip_flag = np.random.randint(0, 2, this_batch_size)
        
        images = (Image.open(p) for p in batch_image_path_list)
        images = (ImageOps.mirror(im) if f else im for im, f in zip(images, flip_flag))
        images = (im.rotate(k * 90) for im, k in zip(images, rotate_angle))
        images = [np.asarray(im)[:,:,0:3]  for im in images]
        
        images = np.stack(images)
        images = images.transpose(0, 3, 1, 2)
        images = images.astype(np.float32) / 255.0
        images = np.ascontiguousarray(images)
        
        labels = (rotate_angle == 0)
        labels = np.stack([labels, np.logical_not(labels)]).transpose()
        labels = labels.astype(np.float32)
        labels = np.ascontiguousarray(labels)
        
        input_map = {x_pl: images, y_pl: labels}
        _, outputs = trainer.train_minibatch(input_map, outputs=(loss, acc))
        
        sum_loss += outputs[loss].sum()
        sum_acc += len(batch_image_path_list) - outputs[acc].sum()
        sum_seen += len(batch_image_path_list)
        
        my_bar.set_description(f"loss: {sum_loss/sum_seen:0.6f}, acc: {sum_acc/sum_seen:0.6f}")

    print(f'e: {epoch},\t loss: {sum_loss/sum_seen},\t acc: {sum_acc/sum_seen}')
    with open(logfile_path, "a") as fout:
        fout.write(f"t, {epoch}, {sum_loss/sum_seen}, {sum_acc/sum_seen}\n")

    if epoch%10 == 0:
        image_path_list = test_image_path_list
        
        sum_loss = 0.0
        sum_acc = 0.0
        sum_seen = 0.0
        
        my_bar = tqdm(range(0, len(image_path_list), test_batch_size), leave=False)
        for i in my_bar:
            batch_image_path_list = image_path_list[i:i+test_batch_size]
            this_batch_size = len(batch_image_path_list)
            rotate_angle = np.array([0, 1, 2, 3, 0, 1, 2, 3] * this_batch_size, dtype=np.int8)
            flip_flag = np.array([0, 0, 0, 0, 1, 1, 1, 1] * this_batch_size, dtype=np.int8)

            images = ([Image.open(p)]*8 for p in batch_image_path_list)
            images = itertools.chain.from_iterable(images)
            images = (ImageOps.mirror(im) if f else im for im, f in zip(images, flip_flag))
            images = (im.rotate(k * 90) for im, k in zip(images, rotate_angle))
            images = [np.asarray(im)[:,:,0:3]  for im in images]
            
            images = np.stack(images)
            images = images.transpose(0, 3, 1, 2)
            images = images.astype(np.float32) / 255.0
            images = np.ascontiguousarray(images)
            
            labels = (rotate_angle == 0)
            labels = np.stack([labels, np.logical_not(labels)]).transpose()
            labels = labels.astype(np.float32)
            labels = np.ascontiguousarray(labels)
            
            input_map = {x_pl: images, y_pl: labels}
            batch_loss = loss.eval(input_map)
            batch_acc = acc.eval(input_map)
            
            sum_loss += batch_loss.sum()
            sum_acc += len(batch_image_path_list) * 8 - batch_acc.sum()
            sum_seen += len(batch_image_path_list) * 8
            
            my_bar.set_description(f"loss: {sum_loss / sum_seen:0.6f}, acc: {sum_acc / sum_seen:0.6f}")
            
        print(f'test e: {epoch},\t loss: {sum_loss/sum_seen},\t acc: {sum_acc/sum_seen}')
        with open(logfile_path, "a") as fout:
            fout.write(f"e, {epoch}, {sum_loss/sum_seen}, {sum_acc/sum_seen}\n")

    if epoch%10 == 0:
        model_save_path = log_dir / f"model_{epoch:06}.dat"
        model_save_path_onnx = log_dir / f"model_{epoch:06}.onnx"
        
        trainer.save_checkpoint(str(model_save_path))
        model.save(str(model_save_path_onnx), format=C.ModelFormat.ONNX)
