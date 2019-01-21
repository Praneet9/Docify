import cv2
import pytesseract as pyt
import re
from ctpn.demo_pb import get_coords
import numpy as np
from keras.models import model_from_json
import tensorflow as tf
from keras import backend as k

def recognise_text(image_path, photo_path):
    
    image = cv2.imread(image_path, 0)

    coordinates = get_coords(image_path)

    detected_text = []

    coordinates = sorted(coordinates, key = lambda coords: coords[1])

    for coords in coordinates:
        x, y, w, h = coords
        temp = image[y:h, x:w]

        _, thresh = cv2.threshold(temp, 100, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        thresh = cv2.copyMakeBorder(thresh, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=[255, 255, 255])
        text = pyt.image_to_string(thresh, lang="eng+hin+mar", config=('--oem 1 --psm 3'))
        
        text = clean_text(text)
        if len(text) < 3:
            continue
        detected_text.append(text)

    face, found = get_photo(image)

    if found:
        cv2.imwrite(photo_path, face)
    else:
        photo_path = face
    
    return detected_text, photo_path


def clean_text(text):
    if text != ' ' or text != '  ' or text != '':
        text = re.sub('[^A-Za-z0-9-/ ]+', '', text)
        text = text.lstrip()
        text = text.rstrip()
        text = re.sub(r'\s{2,}', ' ', text) 
        
    return text

def get_labels_from_licence(details1):
    imp = {}
    for idx in range(len(details1)):
        if 'DL No' in details1[idx]:
            try:
                imp["DL NO"] = details1[idx].split('DL No')[-1].strip()
            except Exception as _:
                imp["DL NO"] = "Not Found"
            del details1[idx]
        elif details1[idx].startswith('DOB'):
            dob = re.findall(r"([0-9]{2}\-[0-9]{2}\-[0-9]{4})", details1[idx].split(' ', 1)[-1])[0]
            #imp["DOB"] = details1[idx].split(' ', 1)[1].strip().split(r"[0-9]{0,2}\-[0-9]{0,2}\-[0-9]{0,4}", 1)[0]
            imp["Date of Birth"] = dob
            del details1[idx]
            imp["Name"] = details1[idx + 1].split(' ', 1)[-1].strip()
            del details1[idx + 1]
            try:
                imp["Father's Name"] = details1[idx + 2].split('of',1)[1].strip()
                del details1[idx + 2]
            except Exception as _:
                imp["Father's Name"] = details1[idx + 2].split('Of',1)[1].strip()
                del details1[idx + 2]
            i = 4
            address = details1[idx + 3].split('Add', 1)[1].strip()
            del details1[idx + 3]
            while not details1[idx + i].startswith('PIN') and i < 8:
                if details1[idx + i].isupper() != True:
                    del details1[idx + i]
                    i += 1
                    continue
                address += ' ' + details1[idx + i]
                del details1[idx + i]
                i += 1
            imp["Address"] = address
            imp["Pin Code"] = details1[idx + i].split(' ', 1)[1]
            del details1[idx + i]
            break
        elif details1[idx].startswith('Name'):
            dob = re.findall(r"([0-9]{2}\-[0-9]{2}\-[0-9]{4})", details1[idx - 1].split(' ', 1)[1])[0]
            imp["Date of Birth"] = dob
            del details1[idx - 1]
            imp["Name"] = details1[idx][4:].strip()
            del details1[idx]
            try:
                imp["Father's Name"] = details1[idx + 2].split('of',1)[1].strip()
                del details1[idx + 2]
            except Exception as _:
                imp["Father's Name"] = details1[idx + 2].split('Of',1)[1].strip()
                del details1[idx + 2]
            i = 3
            address = details1[idx + 2].split('Add', 1)[1].strip()
            del details1[idx + 2]
            while not details1[idx + i].startswith('PIN') and i < 7:
                if details1[idx + i].isupper() != True:
                    del details1[idx + i]
                    i += 1
                    continue
                address += ' ' + details1[idx + i]
                del details1[idx + i]
                i += 1
            imp["Address"] = address
            imp["Pin Code"] = details1[idx + i].split(' ', 1)[1]
            del details1[idx + i]
            break
    return imp


def get_labels_from_aadhar(temp):
    imp = {}
    temp = temp[::-1]
    for idx in range(len(temp)):
        if re.search(r"[0-9]{4}\s[0-9]{4}\s[0-9]{4}", temp[idx]):
            try:
                imp['Aadhar No'] = re.findall(r"[0-9]{4}\s[0-9]{4}\s[0-9]{4}", temp[idx])[0]
            except Exception as _:
                imp['Aadhar No'] = "Not Found"
            if temp[idx + 1].endswith("Female") or temp[idx + 1].endswith("FEMALE"):
                imp["Gender"] = "Female"
            elif temp[idx + 1].endswith("Male") or temp[idx + 1].endswith("MALE"):
                imp["Gender"] = "Male"
            elif temp[idx + 2].endswith("Female") or temp[idx + 2].endswith("FEMALE"):
                imp["Gender"] = "Female"
            elif temp[idx + 2].endswith("Male") or temp[idx + 2].endswith("MALE"):
                imp["Gender"] = "Male"
            elif temp[idx + 3].endswith("Female") or temp[idx + 3].endswith("FEMALE"):
                imp["Gender"] = "Female"
            elif temp[idx + 3].endswith("Male") or temp[idx + 3].endswith("MALE"):
                imp["Gender"] = "Male"
        elif re.search(r"[0-9]{2}\-|/[0-9]{2}\-|/[0-9]{4}", temp[idx]):
            try:
                imp["Date of Birth"] = re.findall(r"[0-9]{2}\-[0-9]{2}\-[0-9]{4}", temp[idx])[0]
            except Exception as _:
                imp["Date of Birth"] = re.findall(r"[0-9]{2}/[0-9]{2}/[0-9]{4}", temp[idx])[0]
            imp["Name"] = temp[idx + 1]
        elif "Year of Birth" in temp[idx]:
            try:
                imp["Year of Birth"] = re.findall(r"[0-9]{4}", temp[idx])[0]
            except Exception as _:
                imp["Year of Birth"] = "Not Found"
            imp["Name"] = temp[idx + 1]
        elif re.search(r"[0-9]{4}", temp[idx]):
            try:
                imp["Year of Birth"] = re.findall(r"[0-9]{4}", temp[idx])[0]
            except Exception as _:
                imp["Year of Birth"] = "Not Found"
            imp["Name"] = temp[idx + 1]
        elif len(temp[idx].split(' ')) > 2:
            if 'GOVERNMENT' in temp[idx] or 'OF' in temp[idx] or 'INDIA' in temp[idx]:
                continue
            else:
                imp["Name"] = temp[idx]
    return imp


def seven_segment(image_path):
    
    image = cv2.imread(image_path, 0)

    hist, _ = np.histogram(image,256,[0,256])

    _, img = cv2.threshold(image, np.argmax(hist) - 15, 255, cv2.THRESH_BINARY_INV)

    noise_cleared = cv2.fastNlMeansDenoising(img, None, 4, 7, 21)

    lines_removed = _lineRemoval(noise_cleared)

    text = _character_segmentation(lines_removed)

    return text


def get_photo(image):
    '''
    Image Should be 1920 x 1080 pixels
    '''
    scale_factor = 1.1
    min_neighbors = 3
    min_size = (150, 150)
    flags = cv2.CASCADE_SCALE_IMAGE

    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    faces = face_cascade.detectMultiScale(image, scaleFactor = scale_factor, minNeighbors = min_neighbors,
                                          minSize = min_size, flags = flags)
    
    try:
        x, y, w, h = faces[0]
        face = image[y-50:y+h+40, x-10:x+w+10]
        return face, True
    except Exception as _:
        return "Photo not found!", False

def _lineRemoval(img):
    min_length=140
    matrix = _imgToMatrixR(img)
    for i in range(0, len(matrix)):
        row=matrix[i]
        start=-1
        end=0
        conn=0
        for j in range(0, len(row)):
            if (row[j]==0):
                conn=conn+1
                # first point in the line .
                if( start == -1 ):
                    start = j
                # last point in the row .
                if( j == len(row)-1 ):
                    end =j
                    if (conn > min_length):
                        img[i-2:i+4, start:end+1] = 255
                    start = -1
                    end = 0
                    conn = 0
            # end of the line
            else:
                end =j
                if (conn >min_length):
                    img[i-2:i+4, start:end+1] = 255
                start = -1
                end = 0
                conn = 0
#     showImage('after line', img)
    return img

'''
this function convert image into matrix of image rows
'''
def _imgToMatrixR(img):
    # get dimensions
    height, width = img.shape
    matrix = []
    # getting pixels values for all rows
    for i in range(0, height):
        row = []
        for j in range(0, width):
            row.append(img[i,j])
        matrix.append(row)
    return matrix

'''
this function convert image into matrix of image columns
'''
def _imgToMatrixC(img):
    # get dimensions
    height, width = img.shape
    matrix = []
    # getting pixels values for all columns
    for i in range(0, width):
        col = []
        for j in range(0, height):
            col.append(img[j, i])
        matrix.append(col)
    return matrix

'''
this function clears all horizontal boundaries around the input image
'''
def clearBounds_horiz(img):
#     showImage('before horizontal', img)
    height, width = img.shape
    matrix = _imgToMatrixR(img)
    white_counter = _countPixel(matrix,255)

    for i in range (0,height):
        if(white_counter[i]>= width-1):
            img = img[1:height,0:width]
        else:
            break

    new_height, width = img.shape
    for i in range (1,height):
        if(white_counter[height-i]>= width-1):
            img = img[0:new_height-i,0:width]
        else:
            break
#     showImage('after horizontal', img)
    return img

'''
this function clears all vertical boundaries around the input image
'''
def clearBounds_vert(img):
#     showImage('before vertical', img)
    height, width = img.shape
    matrix = _imgToMatrixC(img)
    white_counter = _countPixel(matrix,255)

    for i in range (0,width):
        if(white_counter[i]>= height-1):
            img = img[0:height,1:width]
        else:
            break

    height, new_width = img.shape
    for i in range (1,width):
        if(white_counter[width-i]>= height-1):
            img = img[0:height,0:new_width-i]
        else:
            break
#     showImage('after vertical', img)
    return img

'''
this function count a specific value (parameter p) in matrix
'''
def _countPixel(matrix,p):
    counter = []
    for k in range(0, len(matrix)):
        counter.append(matrix[k].count(p))
    return counter

def _character_segmentation(img):
    height = img.shape[0] / 3
    
    dilated = cv2.dilate(img, np.ones((1, 1)), iterations = 2)
    dilated = cv2.dilate(img, np.ones((40, 1)), iterations = 1)
    
    # canny = cv2.Canny(dilated, 30, 150)
    
    _, ctrs_line, _ = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    sorted_ctrs_line = sorted(ctrs_line, key=lambda ctr: cv2.boundingRect(ctr)[0])
    
    text = ''

    for ctr_line in sorted_ctrs_line:
        x_character, y_character, w_character, h_character = cv2.boundingRect(ctr_line)
        if h_character < height:
            continue

        cropped_line = img[y_character:y_character + h_character, x_character:x_character + w_character]
#         cropped_line = cv2.copyMakeBorder(cropped_line, 2, 2, 2, 2, cv2.BORDER_CONSTANT)
#         showImage('image', cropped_line)
        
        cropped_line = cv2.resize(cropped_line, (20, 20), None)
        cropped_line = cv2.copyMakeBorder(cropped_line, 6, 6, 6, 6, cv2.BORDER_CONSTANT)

        k.set_session(session)
        with my_graph.as_default():
            output = model.predict_classes(cropped_line.reshape(-1, 32, 32, 1))
            text += str(output[0])

    return text

def _init_model():
    global model, my_graph, session

    my_graph = tf.Graph()
    with my_graph.as_default():
        session = tf.Session()
        with session.as_default():
            json_file = open('model/model.json', 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            model = model_from_json(loaded_model_json)
            model.load_weights("model/model.h5")
            print("Loaded Model from disk")

            # compile and evaluate model
            model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])