import cv2
import face_recognition
import numpy as np

def detect_faces(image_path):
    faceDetector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    image = cv2.imread(image_path)    
    faces = faceDetector.detectMultiScale(image, scaleFactor=1.1, minNeighbors = 3, minSize = (250,250), flags = cv2.CASCADE_SCALE_IMAGE)
    
    try:
        x,y,w,h = faces[0]
    except:
        return 'No Face Found'
    face = image[y-50:y+h+40, x-10:x+w+10]
    return face

def match_faces(id_card_image, ref_image):
    id_card = detect_faces(id_card_image)
    ref = detect_faces(ref_image)
    try:
        ref = cv2.resize(ref, (id_card.shape[1], id_card.shape[0]))

        id_card_encodings = face_recognition.face_encodings(id_card)[0]
        ref_encodings = face_recognition.face_encodings(ref)[0]

        result = face_recognition.compare_faces([id_card_encodings], ref_encodings)[0]
        percent = face_recognition.face_distance([id_card_encodings], ref_encodings)[0]
        percent = (1 - percent) * 100.00

        return result, percent
    except:
        return False, 0