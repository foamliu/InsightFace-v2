import cv2 as cv
import numpy as np
from PIL import Image

from mtcnn.detector import detect_faces
from mtcnn.visualization_utils import show_bboxes

if __name__ == '__main__':
    img = Image.open('images/office1.jpg')
    bounding_boxes, landmarks = detect_faces(img)
    img = cv.cvtColor(np.array(img), cv.COLOR_RGB2BGR)
    show_bboxes(img, bounding_boxes, landmarks)

    img = Image.open('images/office2.jpg')
    bounding_boxes, landmarks = detect_faces(img)
    img = cv.cvtColor(np.array(img), cv.COLOR_RGB2BGR)
    show_bboxes(img, bounding_boxes, landmarks)

    img = Image.open('images/office3.jpg')
    bounding_boxes, landmarks = detect_faces(img)
    img = cv.cvtColor(np.array(img), cv.COLOR_RGB2BGR)
    show_bboxes(img, bounding_boxes, landmarks)

    img = Image.open('images/office4.jpg')
    bounding_boxes, landmarks = detect_faces(img)
    img = cv.cvtColor(np.array(img), cv.COLOR_RGB2BGR)
    show_bboxes(img, bounding_boxes, landmarks)

    img = Image.open('images/office5.jpg')
    bounding_boxes, landmarks = detect_faces(img)
    img = cv.cvtColor(np.array(img), cv.COLOR_RGB2BGR)
    show_bboxes(img, bounding_boxes, landmarks)
