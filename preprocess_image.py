import cv2
import numpy as np
import torch

mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]

def process_image(image_path):
  global mean, std
  image = cv2.imread(image_path, cv2.IMREAD_COLOR)
  image = image.astype(np.float32)[:, :, ::-1]
  image = image / 255.0
  image -= mean
  image /= std
  image = image.transpose((2, 0, 1))
  return torch.from_numpy(image)
  
