import cv2
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms

mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]

to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
])

def process_image(image_path, using_torchvision=False):
  if using_torchvision:
    return _torch_process(image_path)
  else:
    return _normal_process(image_path)
  
def _normal_process(image_path)
  global mean, std
  origin_image = cv2.imread(image_path, cv2.IMREAD_COLOR)
  image = origin_image.astype(np.float32)[:, :, ::-1]
  image = image / 255.0
  image -= mean
  image /= std
  image = image.transpose((2, 0, 1))
  return origin_image, torch.from_numpy(image)
  
def _torch_process(image_path):
  global to_tensor
  origin_image = cv2.imread(image_path, cv2.IMREAD_COLOR)
  img = Image.open(image_path)
  return origin_image, to_tensor(img)
