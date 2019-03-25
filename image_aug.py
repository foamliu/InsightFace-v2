import os
import pickle
import random

import cv2 as cv
import numpy as np
from PIL import Image
from imgaug import augmenters as iaa
from torchvision import transforms

from config import IMG_DIR
from config import pickle_file

# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        transforms.ColorJitter(brightness=0.6, contrast=0.6, saturation=0.6, hue=0.2),
    ]),
}
transformer = data_transforms['train']
sometimes = lambda aug: iaa.Sometimes(0.5, aug)

# Define our sequence of augmentation steps that will be applied to every image.
seq = iaa.Sequential(
    [
        iaa.Fliplr(0.5),  # horizontally flip 50% of all images

        iaa.SomeOf((0, 2),
                   [
                       iaa.OneOf([
                           iaa.GaussianBlur((0, 0.5)),
                           iaa.AverageBlur(k=(2, 3)),
                           iaa.MedianBlur(k=(3, 4)),
                       ]),

                       sometimes(iaa.OneOf([
                           iaa.EdgeDetect(alpha=(0, 0.7)),
                           iaa.DirectedEdgeDetect(
                               alpha=(0, 0.7), direction=(0.0, 1.0)
                           ),
                       ])),

                       iaa.AdditiveGaussianNoise(
                           loc=0, scale=(0.0, 0.01 * 255), per_channel=0.5
                       ),

                       iaa.OneOf([
                           iaa.Dropout((0.01, 0.02), per_channel=0.5),
                           iaa.CoarseDropout(
                               (0.03, 0.15), size_percent=(0.01, 0.02),
                               per_channel=0.2
                           ),
                       ]),

                       # Add a value of -10 to 10 to each pixel.
                       iaa.Add((-5, 5), per_channel=0.5),

                       # Change brightness of images (50-150% of original value).
                       iaa.Multiply((0.9, 1.1), per_channel=0.5),

                       # Improve or worsen the contrast of images.
                       iaa.ContrastNormalization((0.9, 1.1), per_channel=0.5),

                       iaa.Grayscale(alpha=1.0),
                   ],
                   # do all of the above augmentations in random order
                   random_order=True
                   )
    ]
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
    img = cv.imread(filename)  # BGR
    cv.imwrite('origin.png', img)
    img = img[..., ::-1]  # RGB
    img = Image.fromarray(img, 'RGB')  # RGB
    img = transformer(img)
    img = image_aug(img)  # RGB
    img = img[..., ::-1]  # BGR
    cv.imwrite('out.png', img)
