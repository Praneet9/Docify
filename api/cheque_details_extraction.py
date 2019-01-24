## All neccessary imports ##
import cv2
import re
import imutils
import numpy as np
import pytesseract as pyt
from imutils import contours
from skimage.segmentation import clear_border


## New MICR Method ##
def get_micrcode(image_name):
    try:
        image = cv2.imread(image_name, 0)
        image = cv2.resize(image, (1920,1080))

        (h,w,) = image.shape[:2]
        delta = int(h - (h*0.17))
        bottom = image[delta:h, 0:w]

        thresh = cv2.threshold(bottom, 100, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

        text = pyt.image_to_string(thresh, lang='mcr', config='--oem 1 --psm 3')

        a, b, c, d = text.split()[:4]

        if len(b) > 10:
            b = b[0:9]
            b += 'a'
        return a + ' ' + b + ' ' + c + ' ' + d
    except:
        return 'MICR Not Found'
## New MICR End ##

#### IFSC #####
def get_ifsc(image_path):
    
    def replace(text):
        # Remove some noise present in the text
        chars = "`*_{}[]()>#+-.!$:;?"
        for c in chars:
            text = text.replace(c, '')
        return text
    
    # Read image
    image = cv2.imread(image_path)
    image = cv2.resize(image, (1920,1080))
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    luminance, a, b = cv2.split(lab)
    
    hist,bins = np.histogram(luminance,256,[0,256])

    mean = int((np.argmax(hist) + np.argmin(hist)) / 2)

    luminance[luminance > mean] = 255
    luminance[luminance <= mean] = 0
    
    # Forward it to ocr to get all the text present in image
    text = pyt.image_to_string(luminance, config=('--oem 1 --psm 3'))
    
    # Find IFSC in text and find the IFSC Code using regex
    ifsc = text.find('IFSC')
    # Select the range where the real IFSC Code will be present
    text = text[ifsc: ifsc + 30]
    
    text = replace(text)
    try:
        text = re.findall(r'[A-Z0-9]{11}', text)[0]
    except:
        return 0
    return text

def get_ifsc2(image_path):
    
    def replace(text):
        # Remove some noise present in the text
        chars = "`*_{}[]()>#+-.!$:;?"
        for c in chars:
            text = text.replace(c, '')
        return text
    
    # Read image
    image = cv2.imread(image_path)
    image = cv2.resize(image, (1920,1080))
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    luminance, a, b = cv2.split(lab)
    
    hist,bins = np.histogram(luminance,256,[0,256])

    mean = int((np.argmax(hist) + np.argmin(hist)) / 2)

    luminance[luminance > mean] = 255
    luminance[luminance <= mean] = 0
    
    # Read template
    template = cv2.imread('templates/template_ifsc.png')
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    template_thresh = cv2.threshold(template_gray, 150, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    
    diff = cv2.subtract(luminance, template_thresh)
    diff = cv2.bitwise_and(diff, gray_image)
    # Forward it to ocr to get all the text present in image
    text = pyt.image_to_string(diff, config=('--oem 1 --psm 3'))
    
    # Find IFSC in text and find the IFSC Code using regex
    
    # Select the range where the real IFSC Code will be present
    text = replace(text)
    try:
        text = re.findall(r'[A-Z0-9]{11}', text)[0]
    except:
        return 0
    return text

def get_ifsc3(image):
    
    def replace(text):
        return text.replace('?', '7')
    
    img = cv2.imread(image)
    text = pyt.image_to_string(img, config=('--oem 1 --psm 3'))
    
    ifsc = text.find('IFSC')
    new_text = text[ifsc : ifsc + 30]
    new_text = replace(new_text)
    
    try:
        code = re.findall(r'[A-Z0-9]{11}', new_text)[0]
    except:
        return 0
    return code

def ensemble_ifsc_output(cheque_img):
    ifsc1 = get_ifsc(cheque_img)
    ifsc2 = get_ifsc2(cheque_img)
    ifsc3 = get_ifsc3(cheque_img)
    ifsc = [ifsc1, ifsc2, ifsc3]
    
    if ifsc1 == 0 and ifsc2 == 0 and ifsc3 == 0:
        return 'IFSC Not Found'
    else:
        for code in ifsc:
            if code != 0:
                return code
        return 'IFSC Not Found'
    
#### IFSC END #####


#### Account No ####
def get_acc(image_path):
    # Read image
    image = cv2.imread(image_path)
    image = cv2.resize(image, (1920,1080))
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    luminance, a, b = cv2.split(lab)
    
    hist,bins = np.histogram(luminance,256,[0,256])

    mean = int((np.argmax(hist) + np.argmin(hist)) / 2)

    luminance[luminance > mean] = 255
    luminance[luminance <= mean] = 0
    
    # Read template
    template = cv2.imread('templates/template_acc.jpg', 0)
    
    thresh = cv2.threshold(template, 150, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    
    # Set difference
    diff = cv2.subtract(luminance, template)
    
    text = pyt.image_to_string(diff, config=('--oem 1 --psm 3'))
    
    if '-' in list(text):
        
        text = text.replace('-', '')
        
    try:
        acc_no = re.findall(r'[0-9]{9,18}',text)[0]
    except:
        text = pyt.image_to_string(luminance, config=('--oem 1 --psm 3'))
        if '-' in list(text):
            
            text = text.replace('-', '')
        try:
            acc_no = re.findall(r'[0-9]{9,18}',text)[0]
        except:
            return 0
    return acc_no
    
def get_acc2(cheque_img):
    img = cv2.imread(cheque_img)
    
    text = pyt.image_to_string(img, config=('--oem 1 --psm 3'))
    
    if '-' in list(text):
        text = text.replace('-', '')
    try:
        text = re.findall(r'[0-9]{9,18}', text)[0]
    except:
        return 0
    return text


def ensemble_acc_output(cheque_img):
    acc1 = get_acc(cheque_img)
    acc2 = get_acc2(cheque_img)
    acc = [acc1, acc2]
    
    
    if acc1 == 0 and acc2 == 0:
        return 'Account Number Not Found'
    else:
        for no in acc:
            if no != 0:
                return no
        return 'Account Number Not Found'
#### Account No END ####