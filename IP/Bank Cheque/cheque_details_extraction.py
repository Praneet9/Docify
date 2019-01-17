import cv2
import re
import imutils
import numpy as np
import pytesseract as pyt
from imutils import contours
from skimage.segmentation import clear_border


def extract_digits_and_symbols(image, charCnts, minW=5, minH=15):
    # grab the internal Python iterator for the list of character
    # contours, then  initialize the character ROI and location
    # lists, respectively
    charIter = charCnts.__iter__()
    rois = []
    locs = []

    # keep looping over the character contours until we reach the end
    # of the list
    while True:
        try:
            # grab the next character contour from the list, compute
            # its bounding box, and initialize the ROI
            c = next(charIter)
            (cX, cY, cW, cH) = cv2.boundingRect(c)
            roi = None

            # check to see if the width and height are sufficiently
            # large, indicating that we have found a digit
            if cW >= minW and cH >= minH:
                # extract the ROI
                roi = image[cY:cY + cH, cX:cX + cW]
                rois.append(roi)
                locs.append((cX, cY, cX + cW, cY + cH))

            # otherwise, we are examining one of the special symbols
            else:
                # MICR symbols include three separate parts, so we
                # need to grab the next two parts from our iterator,
                # followed by initializing the bounding box
                # coordinates for the symbol
                parts = [c, next(charIter), next(charIter)]
                (sXA, sYA, sXB, sYB) = (np.inf, np.inf, -np.inf,-np.inf)

                # loop over the parts
                for p in parts:
                    # compute the bounding box for the part, then
                    # update our bookkeeping variables
                    (pX, pY, pW, pH) = cv2.boundingRect(p)
                    sXA = min(sXA, pX)
                    sYA = min(sYA, pY)
                    sXB = max(sXB, pX + pW)
                    sYB = max(sYB, pY + pH)

                # extract the ROI
                roi = image[sYA:sYB, sXA:sXB]
                rois.append(roi)
                locs.append((sXA, sYA, sXB, sYB))

        # we have reached the end of the iterator; gracefully break
        # from the loop
        except StopIteration:
            break
    # return a tuple of the ROIs and locations
    return (rois, locs)

def detect_micr(cheque_image):
    reference_image = 'ref.png'

    # initialize the list of reference character names, in the same
    # order as they appear in the reference image where the digits
    # their names and:
    # T = Transit (delimit bank branch routing transit #)
    # U = On-us (delimit customer account number)
    # A = Amount (delimit transaction amount)
    # D = Dash (delimit parts of numbers, such as routing or account)
    charNames = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0",
        "T", "U", "A", "D"]

    # load the reference MICR image from disk, convert it to grayscale,
    # and threshold it, such that the digits appear as *white* on a
    # *black* background
    ref = cv2.imread(reference_image)
    # ref = cv2.resize(ref, (1920,1080))
    ref = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
    ref = imutils.resize(ref, width=500)
    ref = cv2.threshold(ref, 0, 255, cv2.THRESH_BINARY_INV |
        cv2.THRESH_OTSU)[1]

    # find contours in the MICR image (i.e,. the outlines of the
    # characters) and sort them from left to right
    refCnts = cv2.findContours(ref.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
    refCnts = refCnts[0] if imutils.is_cv2() else refCnts[1]
    refCnts = contours.sort_contours(refCnts, method="left-to-right")[0]

    # extract the digits and symbols from the list of contours, then
    # initialize a dictionary to map the character name to the ROI
    refROIs = extract_digits_and_symbols(ref, refCnts,
        minW=10, minH=20)[0]
    chars = {}

    # loop over the reference ROIs
    for (name, roi) in zip(charNames, refROIs):
        # resize the ROI to a fixed size, then update the characters
        # dictionary, mapping the character name to the ROI
        roi = cv2.resize(roi, (36, 36)) 
        chars[name] = roi

    # initialize a rectangular kernel (wider than it is tall) along with
    # an empty list to store the output of the check OCR
    rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (24, 8))
    output = []

    
    # load the input image, grab its dimensions, and apply array slicing
    # to keep only the bottom 18% of the image (that's where the account
    # information is)
    image = cv2.imread(cheque_image)
    image = cv2.resize(image, (1920,1080))
    (h, w,) = image.shape[:2]
    delta = int(h - (h * 0.18))
    bottom = image[delta:h, 0:w]

    # convert the bottom image to grayscale, then apply a blackhat
    # morphological operator to find dark regions against a light
    # background (i.e., the routing and account numbers)
    gray = cv2.cvtColor(bottom, cv2.COLOR_BGR2GRAY)
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rectKernel)

    # compute the Scharr gradient of the blackhat image, then scale
    # the rest back into the range [0, 255]
    gradX = cv2.Sobel(blackhat, ddepth=cv2.CV_32F, dx=1, dy=0,
        ksize=-1)
    gradX = np.absolute(gradX)
    (minVal, maxVal) = (np.min(gradX), np.max(gradX))
    gradX = (255 * ((gradX - minVal) / (maxVal - minVal)))
    gradX = gradX.astype("uint8")

    # apply a closing operation using the rectangular kernel to help
    # cloes gaps in between rounting and account digits, then apply
    # Otsu's thresholding method to binarize the image
    gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKernel)
    thresh = cv2.threshold(gradX, 0, 255,
        cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    
    # remove any pixels that are touching the borders of the image (this
    # simply helps us in the next step when we prune contours)
    thresh = clear_border(thresh)
    # cv2.imwrite('intermediat_gradx.png', thresh)


    # find contours in the thresholded image, then initialize the
    # list of group locations
    groupCnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
    groupCnts = groupCnts[0] if imutils.is_cv2() else groupCnts[1]
    groupLocs = []

    # loop over the group contours
    for (i, c) in enumerate(groupCnts):
        # compute the bounding box of the contour
        (x, y, w, h) = cv2.boundingRect(c)
        # print('Group Locs' + str(i) + '', (x, y, w, h))
        # only accept the contour region as a grouping of characters if
        # the ROI is sufficiently large
        if w > 38 and h > 10:
            groupLocs.append((x, y, w, h))

    # sort the digit locations from left-to-right
    groupLocs = sorted(groupLocs, key=lambda x:x[0])

    # loop over the group locations
    for (gX, gY, gW, gH) in groupLocs:
        # initialize the group output of characters
        groupOutput = []

        # extract the group ROI of characters from the grayscale
        # image, then apply thresholding to segment the digits from
        # the background of the credit card
        group = gray[gY - 5:gY + gH + 5, gX - 5:gX + gW + 5]
        group = cv2.threshold(group, 0, 255,
            cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

        # cv2.imshow("Group", group)
        # cv2.waitKey(0)

        # find character contours in the group, then sort them from
        # left to right
        charCnts = cv2.findContours(group.copy(), cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)
        charCnts = charCnts[0] if imutils.is_cv2() else charCnts[1]
        charCnts = contours.sort_contours(charCnts,
            method="left-to-right")[0]

        # find the characters and symbols in the group
        (rois, locs) = extract_digits_and_symbols(group, charCnts)

        # loop over the ROIs from the group
        for roi in rois:
            # initialize the list of template matching scores and
            # resize the ROI to a fixed size
            scores = []
            roi = cv2.resize(roi, (36, 36))

            # loop over the reference character name and corresponding
            # ROI
            for charName in charNames:
                # apply correlation-based template matching, take the
                # score, and update the scores list
                result = cv2.matchTemplate(roi, chars[charName],
                    cv2.TM_CCOEFF)
                (_, score, _, _) = cv2.minMaxLoc(result)
                scores.append(score)

            # the classification for the character ROI will be the
            # reference character name with the *largest* template
            # matching score
            groupOutput.append(charNames[np.argmax(scores)])

        # draw (padded) bounding box surrounding the group along with
        # the OCR output of the group
        cv2.rectangle(image, (gX - 10, gY + delta - 10),
            (gX + gW + 10, gY + gY + delta), (0, 0, 255), 2)
        cv2.putText(image, "".join(groupOutput),
            (gX - 10, gY + delta - 25), cv2.FONT_HERSHEY_SIMPLEX,
            0.95, (0, 0, 255), 3)

        # add the group output to the overall check OCR output
        output.append("".join(groupOutput))
        # cv2.imwrite('output.png', image)
    return ' '.join(output)

def get_micrcode(image_name):
    micr_code = detect_micr(image_name)
    a, b, c, d = micr_code.split()
    new_micr = 'U' + re.findall(r'[0-9]{6}', a)[0] + 'U' + ' ' + re.findall(r'[0-9]{9}', b)[0] + 'T' + ' ' + re.findall(r'[0-9]{6}', c)[0] + 'U' + ' ' + re.findall(r'[0-9]{2}', d)[0]
    return new_micr
    
def get_acc(image_path):
    # Read image
    image = cv2.imread(image_path)
    image = cv2.resize(image, (960,540))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img_thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    
    # Read template
    template = cv2.imread('template_acc.jpg', 0)
    template = cv2.resize(template, (960,540))
    thresh = cv2.threshold(template, 150, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    
    # Set difference
    diff = cv2.subtract(gray, template)
    tdiff = cv2.subtract(img_thresh, thresh)
    
    bit = cv2.bitwise_and(diff, tdiff)
    
    #cv2.imwrite('bit.png', bit)
    text = pyt.image_to_string(diff, config=('--oem 1 --psm 3'))
    # print(text)
    if '-' in list(text):
        # print('Inside Loop', text)
        text = text.replace('-', '')
        # print(text)
    try:
        acc_no = re.findall(r'[0-9]{9,18}',text)[0]
    except:
        text = pyt.image_to_string(img_thresh, config=('--oem 1 --psm 3'))
        if '-' in list(text):
            # print('Inside Loop Except', text)
            text = text.replace('-', '')
        acc_no = re.findall(r'[0-9]{9,18}',text)[0]
    return acc_no

def get_ifsc(image_path):
    
    def replace(text):
        # Remove some noise present in the text
        chars = "`*_{}[]()>#+-.!$:;"
        for c in chars:
            text = text.replace(c, '')
        return text
    
    # Read image
    image = cv2.imread(image_path)
    # image = cv2.resize(image, (960,540))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img_thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    
    # Forward it to ocr to get all the text present in image
    text = pyt.image_to_string(img_thresh, config=('--oem 1 --psm 3'))
    # Find IFSC in text and find the IFSC Code using regex
    ifsc = text.find('IFSC')
    # Select the range where the real IFSC Code will be present
    text = text[ifsc: ifsc + 30]
    # print(text)
    text = replace(text)
    try:
        text = re.findall(r'[A-Z0-9]{11}', text)[0]
    except:
        return 'IFSC Not Found'
    return text

image = 'test.jpg'
print('Image Name', image)
print('Detect MICR:-', get_micrcode(image))
print('Detect ACC.No.:-', get_acc(image))
print('Detect IFSC:-', get_ifsc(image))

image = '1.jpg'
print('Image Name', image)
print('Detect MICR:-', get_micrcode(image))
print('Detect ACC.No.:-', get_acc(image))
print('Detect IFSC:-', get_ifsc(image))

image = '2.jpg'
print('Image Name', image)
print('Detect MICR:-', get_micrcode(image))
print('Detect ACC.No.:-', get_acc(image))
print('Detect IFSC:-', get_ifsc(image))

image = '3.jpg'
print('Image Name', image)
print('Detect MICR:-', get_micrcode(image))
print('Detect ACC.No.:-', get_acc(image))
print('Detect IFSC:-', get_ifsc(image))

