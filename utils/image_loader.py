from PIL import Image
from pytorch_grad_cam.utils.image import preprocess_image, deprocess_image
from glob import glob
import random
import torch
import numpy as np
import cv2
import os


def get_image_from_url(url):
    """A function that gets a URL of an image, 
    and returns a numpy image and a preprocessed
    torch tensor ready to pass to the model """

    img = np.array(Image.open(requests.get(url, stream=True).raw))
    rgb_img_float = np.float32(img) / 255
    input_tensor = preprocess_image(rgb_img_float,
                                   mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
    return img, rgb_img_float, input_tensor

def get_image_from_fs(path, resize=None):
    """A function that consumes a filepath of an image, 
    and returns a numpy image and a preprocessed
    torch tensor ready to pass to the model """

    img = np.array(Image.open(path))
    if resize is not None:
        y,x,c = img.shape
        startx = random.randint(0, x - resize[0] - 1)
        starty = random.randint(0, y - resize[1] - 1)
        img = img[starty:starty+resize[1], startx:startx+resize[0], :]
    rgb_img_float = np.float32(img) / 255
    input_tensor = preprocess_image(rgb_img_float,
                                   mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
    return img, rgb_img_float, input_tensor


def load_image_folder_as_tensor(dir_name, max_images=-1, resize=None):
    images = []
    rgb_img_floats = []
    input_tensors = []
    
    files = glob(dir_name + '/*.jpg')
    files.extend(glob(dir_name + '/*.png'))

    for i, filename in enumerate(files):
        if max_images > 0 and i >= max_images:
            break
        img, rgb_img_float, input_tensor = get_image_from_fs(
            filename,
            resize=resize,
        )
        images.append(img)
        rgb_img_floats.append(rgb_img_float)
        input_tensors.append(input_tensor)

    return torch.vstack(input_tensors)