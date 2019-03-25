import os
import pickle
import random

import cv2 as cv
import numpy as np
from imgaug import augmenters as iaa
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

# Define our sequence of augmentation steps that will be applied to every image.
seq = iaa.Sequential(
    [
        iaa.Fliplr(0.5),  # horizontally flip 50% of all images

        iaa.Sometimes(0.5,
                      iaa.Grayscale(alpha=1.0),
                      ),

        # Improve or worsen the contrast of images.
        # iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5),
    ],
    random_order=True
)


def image_aug(src):
    src = np.expand_dims(src, axis=0)
    augs = seq.augment_images(src)
    aug = augs[0]
    return aug


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
    img = image_aug(img)  # RGB
    img = img[..., ::-1]  # BGR
    cv.imwrite('out.png', img)
