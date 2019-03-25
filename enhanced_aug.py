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


def contrast_aug(src, x):
    alpha = 1.0 + random.uniform(-x, x)
    coef = np.array([[[0.299, 0.587, 0.114]]])
    gray = src * coef
    gray = (3.0 * (1.0 - alpha) / gray.size) * np.sum(gray)
    src *= alpha
    src += gray
    return src


def saturation_aug(src, x):
    alpha = 1.0 + random.uniform(-x, x)

    coef = np.array([[[0.299, 0.587, 0.114]]])
    gray = src * coef
    gray = np.sum(gray, axis=2, keepdims=True)
    gray *= (1.0 - alpha)
    src *= alpha
    src += gray
    return src


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
    cv.imwrite('origin.png', img)
    img = img[..., ::-1]  # RGB
    img = Image.fromarray(img, 'RGB')  # RGB
    # img = compress_aug(img)  # RGB
    img = contrast_aug(img, 0.5)  # RGB
    img = transformer(img)  # RGB
    img = np.array(img)
    img = img[..., ::-1]  # BGR
    cv.imwrite('out.png', img)
