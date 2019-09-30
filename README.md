# Docify
Deep Learning based Android app to extract details from Indian ID cards like Aadhar Card, PAN Card and Driving Licence.

#### Tech
Docify uses a number of open source projects to work properly:

* [Tesseract](https://github.com/tesseract-ocr/tesseract) - Tesseract Open Source OCR Engine 
* [Text-Detection-CTPN](https://github.com/eragonruan/text-detection-ctpn/tree/master) - Text detection mainly based on ctpn model in tensorflow
* [Python3.6](https://www.python.org) - duh

# Installation
#### Install Linux Dependencies
```sh
$ sudo apt install cmake
$ sudo apt install tesseract-ocr
$ sudo apt install mongodb
$ sudo apt install libsm6 libxext6
$ sudo apt install supervisor
$ sudo systemctl start mongo
```
#### Download Tesseract Models [ENG+HIN+MAR]
```sh
https://github.com/tesseract-ocr/tessdata_best
https://github.com/BigPino67/Tesseract-MICR-OCR
```

#### Install Python-Dependencies
```sh
$ pip3 install opencv-python easydict flask face_recognition gunicorn tensorflow keras pytesseract dlib imutils opencv-contrib-python pymongo PyYAML scikit-image scikit-learn
```
#### Start Python Api
```sh
python3 server.py
```
