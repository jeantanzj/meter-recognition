#!/usr/bin/python

# Import the modules
import cv2
import imutils
from imutils.perspective import four_point_transform
from sklearn.externals import joblib
from skimage.feature import hog
import numpy as np
import argparse as ap

# Get the path of the training set
parser = ap.ArgumentParser()
parser.add_argument("-i", "--image", help="Path to Image", required="True")
args = vars(parser.parse_args())

def showImage(img, title):
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.imshow(title, img)
    cv2.waitKey(0)

# Read the input image 
im = cv2.imread(args["image"])
im = imutils.resize(im, height = 300)

# Convert to grayscale and apply Gaussian filtering
im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
im_gray = cv2.GaussianBlur(im_gray, (5, 5), 0)
edged = cv2.Canny(im_gray, 5, 64)
#showImage(edged, 'im_gray')

# Threshold the image
ret, im_th = cv2.threshold(edged, 90, 255, cv2.THRESH_BINARY) #cv2.THRESH_BINARY_INV
#showImage(im_th, 'im_th')

# Find contours in the image
_, ctrs, hier = cv2.findContours(im_th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
ctrs = sorted(ctrs, key=cv2.contourArea, reverse=True)
displayCnt = None
 
# loop over the contours
for c in ctrs:
    # approximate the contour
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
 
    # if the contour has four vertices, then we have found
    # the thermostat display
    if len(approx) == 4:
        displayCnt = approx
        break
#print(displayCnt)
warped = four_point_transform(im_gray, displayCnt.reshape(4, 2))
output = four_point_transform(im, displayCnt.reshape(4, 2))
showImage(warped, 'warped')

th = 0
for th in range(40,55,2): ###CHANGE THESE VALUES
    print(th)
    thresh = cv2.threshold(warped, th, 255,  cv2.THRESH_BINARY_INV)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    showImage(thresh, 'thresh')
