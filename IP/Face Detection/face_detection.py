import cv2 
import numpy as np

def get_photo(image):
    '''
    Image Should be 1920 x 1080 pixels
    '''
    scale_factor = 1.1
    min_neighbors = 3
    min_size = (250, 250)
    flags = cv2.CASCADE_SCALE_IMAGE

    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    image = cv2.imread(image)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(image, scaleFactor = scale_factor, minNeighbors = min_neighbors,
                                          minSize = min_size, flags = flags)
    x, y, w, h = faces[0]
    face = image[y-50:y+h+40, x-10:x+w+10]
	
    return face
