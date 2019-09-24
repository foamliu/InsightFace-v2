import cv2 as cv

from utils import image_aug, get_central_face_attributes, align_face

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
