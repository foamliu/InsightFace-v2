import cv2 as cv
import numpy as np
from imgaug import augmenters as iaa
from torchvision import transforms
from utils import get_central_face_attributes, align_face

# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
    ]),
}
transformer = data_transforms['train']

# Define our sequence of augmentation steps that will be applied to every image.
seq = iaa.Sequential(
    [
        iaa.GaussianBlur(sigma=0.5)
    ]
)


def image_aug(src):
    src = np.expand_dims(src, axis=0)
    augs = seq.augment_images(src)
    aug = augs[0]
    return aug


if __name__ == "__main__":
    filename = 'data/lfw_funneled/Aaron_Eckhart/Aaron_Eckhart_0001.jpg'
    print(filename)
    img = cv.imread(filename)  # BGR
    cv.imshow('', img)
    cv.waitKey(0)

    is_valid, bounding_boxes, landmarks = get_central_face_attributes(filename)
    img = align_face(filename, landmarks)
    cv.imshow('', img)
    cv.waitKey(0)

    img = image_aug(img)  # RGB
    cv.imshow('', img)
    cv.waitKey(0)

    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    cv.imshow('', img)
    cv.waitKey(0)
