#!/usr/bin/python

# Import the modules
import cv2
import imutils
from sklearn.externals import joblib
from skimage.feature import hog
import numpy as np
import argparse as ap

# Get the path of the training set
parser = ap.ArgumentParser()
parser.add_argument("-c", "--classiferPath", help="Path to Classifier File", default="digits_cls.pkl")
parser.add_argument("-i", "--image", help="Path to Image", required="True")
args = vars(parser.parse_args())

def showImage(img, title):
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.imshow(title, img)
    cv2.waitKey(0)

# Load the classifier
clf, pp = joblib.load(args["classiferPath"])

# Read the input image 
im = cv2.imread(args["image"])
im = imutils.resize(im, height = 300)

# Convert to grayscale and apply Gaussian filtering
im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
im_gray = cv2.GaussianBlur(im_gray, (5, 5), 0)
gradients = None
while gradients != 'q': 
	gradients = raw_input("min gradient, maxgradient\n")
	mingradient, maxgradient = [int(x) for x in gradients.split(',')][0:2]
	edged = cv2.Canny(im_gray, mingradient, maxgradient)
	showImage(edged, 'im_gray')
