import cv2
import pytesseract as pyt
import re
from ctpn.demo_pb import get_coords
import numpy as np
from keras.models import model_from_json
import tensorflow as tf
from keras import backend as k


# function to recognise text from image
def recognise_text(image_path, photo_path):
    
    # read image and convert to grayscale
    image = cv2.imread(image_path, 0)

    # get coordinates of text using ctpn
    coordinates = get_coords(image_path)

    detected_text = []

    # sorting coordinates from top to bottom
    coordinates = sorted(coordinates, key = lambda coords: coords[1])

    # looping through all the text boxes
    for coords in coordinates:
        # x, y, width, height of the text box
        x, y, w, h = coords

        # cropping image based on the coordinates
        temp = image[y:h, x:w]

        # binarizing image
        _, thresh = cv2.threshold(temp, 100, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # padding the image with 10 pixels for better prediction with tesseract
        thresh = cv2.copyMakeBorder(thresh, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=[255, 255, 255])
        
        # get text from the image, lang = english + hindi + marathi, config = use lstm for prediction
        text = pyt.image_to_string(thresh, lang="eng+hin+mar", config=('--oem 1 --psm 3'))
        
        # clean text and remove noise
        text = clean_text(text)

        # ignore text if the length of text is less than 3
        if len(text) < 3:
            continue
        detected_text.append(text)

    # find face in the image
    face, found = get_photo(image)

    # if a face is found save it to faces directory
    if found:
        cv2.imwrite(photo_path, face)
    else:
        photo_path = face
    
    # return detected text and the face path
    return detected_text, photo_path


# function to remove noise and unnecessary characters from string
def clean_text(text):
    if text != ' ' or text != '  ' or text != '':
        text = re.sub('[^A-Za-z0-9-/ ]+', '', text)
        text = text.lstrip()
        text = text.rstrip()
        text = re.sub(r'\s{2,}', ' ', text) 
        
    return text


# extract labels from licence image
def get_labels_from_licence(details1):
    imp = {}
    
    # loop through all the details found line wise
    for idx in range(len(details1)):

        # if DL No is found save it
        if 'DL No' in details1[idx]:
            try:
                imp["DL NO"] = details1[idx].split('DL No')[-1].strip()
            except Exception as _:
                imp["DL NO"] = "Not Found"
                
        # if dob is found, use it as a hook and try finding other details relative to it
        elif details1[idx].startswith('DOB'):
            # extract only dob from the text
            dob = re.findall(r"([0-9]{2}\-[0-9]{2}\-[0-9]{4})", details1[idx].split(' ', 1)[-1])[0]
            
            imp["Date of Birth"] = dob
            
            # next line is always name and father's name
            imp["Name"] = details1[idx + 1].split(' ', 1)[-1].strip()
            
            try:
                # split it from 'of' to get Father's Name
                imp["Father's Name"] = details1[idx + 2].split('of',1)[1].strip()
                
            except Exception as _:
                # handle exception if O is capital in 'of'
                imp["Father's Name"] = details1[idx + 2].split('Of',1)[1].strip()
                
            i = 4
            # split next line from Add for address
            address = details1[idx + 3].split('Add', 1)[1].strip()
            
            # keep appending until PIN code is found or address is of more than 4 lines
            while not details1[idx + i].startswith('PIN') and i < 8:
                if details1[idx + i].isupper() != True:
                    
                    i += 1
                    continue
                address += ' ' + details1[idx + i]
                
                i += 1
            imp["Address"] = address
            try:
                # get only pin code from the string
                imp["Pin Code"] = re.findall(r"([0-9]{6})", details1[idx + i].split(' ', 1)[1])[0]
            except Exception as _:
                pass
                
            break
        # if name is found, use it as a hook and try finding other details relative to it
        elif details1[idx].startswith('Name'):
            # extract only dob from the text
            dob = re.findall(r"([0-9]{2}\-[0-9]{2}\-[0-9]{4})", details1[idx - 1].split(' ', 1)[1])[0]
            imp["Date of Birth"] = dob
            
            imp["Name"] = details1[idx][4:].strip()
            
            # next line is always name and father's name
            try:
                # split it from 'of' to get Father's Name
                imp["Father's Name"] = details1[idx + 2].split('of',1)[1].strip()
                
            except Exception as _:
                # handle exception if O is capital in 'of'
                imp["Father's Name"] = details1[idx + 2].split('Of',1)[1].strip()
                
            i = 3
            # split next line from Add for address
            address = details1[idx + 2].split('Add', 1)[1].strip()
            
            # keep appending until PIN code is found or address is of more than 4 lines
            while not details1[idx + i].startswith('PIN') and i < 7:
                if details1[idx + i].isupper() != True:
                    
                    i += 1
                    continue
                address += ' ' + details1[idx + i]
                
                i += 1
            imp["Address"] = address
            try:
                # get only pin code from the string
                imp["Pin Code"] = re.findall(r"([0-9]{6})", details1[idx + i].split(' ', 1)[1])[0]
            except Exception as _:
                pass
                
            break
    return imp


# extract labels from aadhar image
def get_labels_from_aadhar(temp):
    imp = {}

    # reverse list to parse through it starting from the aadhar number
    temp = temp[::-1]
    # parse through the list
    for idx in range(len(temp)):
        # if string similar to aadhar number is found, use it as a hook to find other details
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
            # if string similar to date is found, use it as a hook to find other details
            try:
                imp["Date of Birth"] = re.findall(r"[0-9]{2}\-[0-9]{2}\-[0-9]{4}", temp[idx])[0]
            except Exception as _:
                imp["Date of Birth"] = re.findall(r"[0-9]{2}/[0-9]{2}/[0-9]{4}", temp[idx])[0]
            imp["Name"] = temp[idx + 1]
        
        elif "Year of Birth" in temp[idx]:
            # handle variation of 'Year of Birth' in place of DOB
            try:
                imp["Year of Birth"] = re.findall(r"[0-9]{4}", temp[idx])[0]
            except Exception as _:
                imp["Year of Birth"] = "Not Found"
            imp["Name"] = temp[idx + 1]
        
        elif re.search(r"[0-9]{4}", temp[idx]):
            # handle exception if Year of Birth is not found but string similar to year is found
            try:
                imp["Year of Birth"] = re.findall(r"[0-9]{4}", temp[idx])[0]
            except Exception as _:
                imp["Year of Birth"] = "Not Found"
            imp["Name"] = temp[idx + 1]
        
        elif len(temp[idx].split(' ')) > 2:
            # following text will be name, ignore line if it includes GOVERNMENT OF INDIA
            if 'GOVERNMENT' in temp[idx] or 'OF' in temp[idx] or 'INDIA' in temp[idx]:
                continue
            else:
                imp["Name"] = temp[idx]
    return imp


def seven_segment(image_path):
    # read image and convert to grayscale
    image = cv2.imread(image_path, 0)
    
    # calculate histogram of the image
    hist, _ = np.histogram(image,256,[0,256])
    
    # binarize image based on the peak value of histogram
    _, img = cv2.threshold(image, np.argmax(hist) - 10, 255, cv2.THRESH_BINARY_INV)
    
    # clear noise if any in the image
    noise_cleared = cv2.fastNlMeansDenoising(img, None, 4, 7, 21)
    
    # remove border lines
    lines_removed = _lineRemoval(noise_cleared)
    
    # segment character by character for prediction
    text = _character_segmentation(lines_removed)

    return text

# function to find face in the image
def get_photo(image):

    # Image Should be 1920 x 1080 pixels
    scale_factor = 1.1
    min_neighbors = 3
    min_size = (150, 150)
    flags = cv2.CASCADE_SCALE_IMAGE

    # using frontal face haar cascade
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    # detect faces of different sizes
    faces = face_cascade.detectMultiScale(image, scaleFactor = scale_factor, minNeighbors = min_neighbors,
                                          minSize = min_size, flags = flags)
    
    # crop the face if found
    try:
        x, y, w, h = faces[0]
        face = image[y-50:y+h+40, x-10:x+w+10]
        return face, True
    except Exception as _:
        return "Photo not found!", False

# function to remove border lines
def _lineRemoval(img):
    min_length=140
    # getting matrix of values in all the rows
    matrix = _imgToMatrixR(img)
    # parsing through the matrix row by row
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
    
    return img


# this function converts image into matrix of image rows
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


# this function convert image into matrix of image columns
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


# this function clears all horizontal boundaries around the input image
def clearBounds_horiz(img):
    
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
    
    return img


# this function clears all vertical boundaries around the input image
def clearBounds_vert(img):
    
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
    
    return img


# this function counts a specific value (parameter p) in matrix
def _countPixel(matrix,p):
    counter = []
    for k in range(0, len(matrix)):
        counter.append(matrix[k].count(p))
    return counter


# function to segment image character by character for seven segment prediction
def _character_segmentation(img):
    height = img.shape[0] / 3
    
    # dilating text to fill the gaps between characters
    dilated = cv2.dilate(img, np.ones((1, 1)), iterations = 2)
    # dilating text vertically
    dilated = cv2.dilate(img, np.ones((40, 1)), iterations = 1)
    
    # canny = cv2.Canny(dilated, 30, 150)
    
    # finding contours in the text
    _, ctrs_line, _ = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # sorting contours horizontally
    sorted_ctrs_line = sorted(ctrs_line, key=lambda ctr: cv2.boundingRect(ctr)[0])
    
    text = ''

    # parsing through all the contours
    for ctr_line in sorted_ctrs_line:
        x_character, y_character, w_character, h_character = cv2.boundingRect(ctr_line)
        if h_character < height:
            continue

        # cropping character
        cropped_line = img[y_character:y_character + h_character, x_character:x_character + w_character]
        
        # resizing and adding padding to final size as 32 x 32
        cropped_line = cv2.resize(cropped_line, (20, 20), None)
        cropped_line = cv2.copyMakeBorder(cropped_line, 6, 6, 6, 6, cv2.BORDER_CONSTANT)

        # using predefind session for prediction
        k.set_session(session)
        with my_graph.as_default():
            output = model.predict_classes(cropped_line.reshape(-1, 32, 32, 1))
            text += str(output[0])

    return text


# function to initialize model for prediction
def _init_model():
    global model, my_graph, session

    # creating graph with session as there are more than one deep learning models running simultaneously
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