import os
import pickle
import random
from io import BytesIO

import cv2 as cv
import numpy as np
from PIL import Image
from torchvision import transforms

from config import IMG_DIR
from config import pickle_file

# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train-enhanced': transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.125, contrast=0.125, saturation=0.125, hue=0),
        # transforms.ToTensor(),
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
}


def compress_aug(img):
    buf = BytesIO()
    q = random.randint(2, 20)
    img.save(buf, format='JPEG', quality=q)
    buf = buf.getvalue()
    img = Image.open(BytesIO(buf))
    return img


if __name__ == "__main__":
    with open(pickle_file, 'rb') as file:
        data = pickle.load(file)

    samples = data
    sample = random.sample(samples, 1)[0]
    filename = sample['img']
    filename = os.path.join(IMG_DIR, filename)
    print(filename)
    transformer = data_transforms['train-enhanced']
    img = cv.imread(filename)  # BGR
    img = img[..., ::-1]  # RGB
    img = Image.fromarray(img, 'RGB')  # RGB
    img = compress_aug(img)  # RGB
    img = transformer(img)  # RGB
    img = np.array(img)
    img = img[..., ::-1]  # BGR
    cv.imshow('image', img)
    cv.waitKey(0)
