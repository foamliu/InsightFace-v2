import cv2 as cv

from utils import image_aug, get_central_face_attributes, align_face

if __name__ == "__main__":
    filename = 'data/lfw_funneled/Aaron_Eckhart/Aaron_Eckhart_0001.jpg'
    print(filename)
    img = cv.imread(filename)  # BGR
    cv.imwrite('1.jpg', img)

    is_valid, bounding_boxes, landmarks = get_central_face_attributes(filename)
    img = align_face(filename, landmarks)
    cv.imwrite('2.jpg', img)

    img = image_aug(img)  # RGB
    cv.imwrite('3.jpg', img)

    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    cv.imwrite('4.jpg', img)
