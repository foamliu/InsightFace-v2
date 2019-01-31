import cv2 as cv
import numpy as np
import torch
from torchvision import transforms

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ]),
    'val': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}
transformer = data_transforms['train']

if __name__ == "__main__":
    img = cv.imread('images/0_fn_0.jpg')
    img = transforms.ToPILImage()(img)
    arr = np.array(img)
    print(arr)
    print(np.max(arr))
    print(np.min(arr))
    print(np.mean(arr))
    print(np.std(arr))

    arr = arr.astype(np.float)
    arr = (arr - 127.5) / 128
    print(arr)
    print(np.max(arr))
    print(np.min(arr))
    print(np.mean(arr))
    print(np.std(arr))

    img = transformer(img)
    print(img)
    print(torch.max(img))
    print(torch.min(img))
    print(torch.mean(img))
    print(torch.std(img))
