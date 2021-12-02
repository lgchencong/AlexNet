import os
from shutil import copy
import random


def mkfile(file):
    if not os.path.exists(file):
        os.makedirs(file)


file = 'data/flower_photos'
flower_class = [cla for cla in os.listdir(file) if ".txt" not in cla]
for cla in flower_class:
    mkfile('data/train/'+cla)
    mkfile('data/val/'+cla)
rate = 0.1
for cla in flower_class:
    cla_path = file + '/' + cla + '/'
    images = os.listdir(cla_path)
    num = len(images)
    eval_images = random.sample(images, k = int(num * rate))
    for index, image in enumerate(images):
        if image in eval_images:
            image_path = cla_path + image
            val_path = 'data/val/' + cla + '/'
            copy(image_path, val_path)
        else:
            image_path = cla_path + image
            train_path = 'data/train/' + cla + '/'
            copy(image_path, train_path)
