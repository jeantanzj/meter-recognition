#!/usr/bin/python

###
# Credits: 
#   https://github.com/bikz05/digit-recognition
#   https://www.pyimagesearch.com/2014/04/21/building-pokedex-python-finding-game-boy-screen-step-4-6/
###

#python performRecognition.py -i photos/gt.jpg -v -r -t 45 -b 100 -o 3
#python performRecognition.py -i photos/home.jpg -v -t 82 -b 0 -o 10

# Import the modules
import cv2
import imutils
from imutils.perspective import four_point_transform
from sklearn.externals import joblib
from skimage.feature import hog
import numpy as np
import argparse as ap
import sys

verbose = False

def showImage(img, title=''):
    if verbose:
        cv2.namedWindow(title, cv2.WINDOW_NORMAL)
        cv2.imshow(title, img)
        cv2.waitKey(0)

def predict(im, args):
    
    im = imutils.resize(im, height = 300)
    # h = min(300, len(im))
    # w = int(float(h)/len(im) * len(im[0]))
    # im= cv2.resize(im,(w, h))
    # Load the classifier
    clf, pp = joblib.load(args["classiferPath"])
    # Convert to grayscale and apply Gaussian filtering
    im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    im_gray = cv2.GaussianBlur(im_gray, (5, 5), 0)
    edged = cv2.Canny(im_gray, 5, 64)
    showImage(edged, 'edged')

    # Threshold the image
    ret, im_th = cv2.threshold(edged, 90, 255, cv2.THRESH_BINARY) #cv2.THRESH_BINARY_INV
    showImage(im_th, 'im_th')

    ###### Get a bird's eye view of the screen only ###### 
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
    if displayCnt is None:
        return None, im
    warped = four_point_transform(im_gray, displayCnt.reshape(4, 2))
    output = four_point_transform(im, displayCnt.reshape(4, 2))
    showImage(warped, 'warped')

    ###### Convert the digits to white digits on black background and remove some noise###### 
    th = args["threshold"]
    mode = cv2.THRESH_BINARY_INV if args["invert"] else cv2.THRESH_BINARY
    thresh = cv2.threshold(warped, th, 255,  mode)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    showImage(thresh, 'thresh')

    ###### Find contours after the noise has been removed ######
    _, ctrs, hier = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    ctrs = sorted(ctrs, key=cv2.contourArea, reverse=True)

    # Get rectangles of each contour
    rects = [cv2.boundingRect(ctr) for ctr in ctrs]
    rects = sorted(rects, key=lambda x: x[1])

    ######  Try to identify rows of rectangles (ie the digits should be aligned horizontally next to each other) ###### 
    currBtm = 0
    currHeight = 0
    rectRows = []
    prior = False
    for i in range(len(rects)):
        left, bottom, length, height = rects[i]
        if (abs(bottom - currBtm) <= args["tolerance"]) and (abs(height - currHeight) <= args["tolerance"]):
            if prior:
                rectRows.append([rects[i-1]])
            if len(rectRows) == 0:
                rectRows.append([rects[i]])
            else:
                rectRows[len(rectRows)-1].append(rects[i])
            prior = False

        else:
            currBtm = bottom
            currHeight = height
            prior = True    

    #If there are at least 'mindigits' rectangles in a row, and each rectangle has at least 'minboxarea', these rectangles could possibly be digits
    #If there are more than one set of rows fulfilling this criteria, pick the set which has the least variance in area
    minVar = sys.maxsize
    selected = None
    for s in rectRows:
        if len(s) >= args["mindigits"]:
            areas = map(lambda y: y[2] * y[3], s)
            var = np.var(areas)
            if(min(areas) > args["minboxarea"]) and var <= minVar:
                minVar = var
                selected = s


    # For each rectangular region, calculate HOG features and predict
    # the digit using Linear SVM.
    #selected = rects
    if selected is None:
        print("No suitable rows found out of all the rects")
        print("Rows of rectangles :" ,rectRows)
        if verbose:
            print("Drawing all rects found")
            for rect in rects:
                (left, bottom, length, height) = rect
                cv2.rectangle(output, (left, bottom), (left + length, bottom + height), (0, 255, 0), 3)
            showImage(output, "All rects")
        else:
            print("All rects:", rects)
        return None, im

    #Sort by the left position
    selected = sorted(selected, key=lambda x: x[0])
    predictions = []
    #print(selected)
    for rect in selected:
        (left, bottom, length, height) = rect
        # Draw the rectangles
        cv2.rectangle(output, (left, bottom), (left + length, bottom + height), (0, 255, 0), 3) 
        #showImage(output, "Output")

        # Make a square region around the digit
        #leng = int(rect[3] * 1.6) #tan 45 deg = height/length
        #pt1 = max( int(rect[1] + rect[3] // 2 - leng // 2),0)
        #pt2 = max( int(rect[0] + rect[2] // 2 - leng // 2) , 0)
        #roi = thresh[pt1:pt1+leng, pt2:pt2+leng]
        #roi = warped[pt1:pt1+leng, pt2:pt2+leng]
        roi = warped[bottom:bottom+height, left:left+length]

        # # Resize the image
        roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
        #roi = cv2.dilate(roi, (3, 3))
        showImage(roi, 'roi')
        # Calculate the HOG features
        roi_hog_fd = hog(roi, orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualise=False)
        roi_hog_fd = pp.transform(np.array([roi_hog_fd], 'float64'))
        nbr = clf.predict(roi_hog_fd)

        predictions.append(int(nbr[0]))

    showImage(output,"Resulting Image with Rectangular ROIs")
    return predictions, im

def run(args):

    # Read the input image 
    im = cv2.imread(args["image"])
    
    return predict(im,args)

def threshold_float(x):
    x = float(x)
    if x < 0.0 or x > 255.0:
        raise argparse.ArgumentTypeError("%r not in range [0, 255]"%(x,))
    return x

if __name__ == "__main__":

    parser = ap.ArgumentParser()
    parser.add_argument("-c", "--classiferPath", help="Path to Classifier File", default="classifiers/digits_cls_svhn_nobw.pkl")
    parser.add_argument("-i", "--image", help="Path to Image", required="True")
    parser.add_argument("-t", "--threshold", help="Threshold to remove noise (0-255)", type=threshold_float, default=0.0)
    parser.add_argument("-r", "--invert", help="Invert the threshold", action="store_true")
    parser.add_argument("-b", "--minboxarea", help="Minimum area of rectangle to be considered digits", type=int, default=0)
    parser.add_argument("-o", "--tolerance", help="Tolerance of difference in length and height of digits", type=int, default=3)
    parser.add_argument("-d", "--mindigits", help="Minimum number of digits", type=int, default=5)
    parser.add_argument("-v", '--verbose', help="Show image at each step", action="store_true")
    args = vars(parser.parse_args())
    print("Args:", args)
    verbose = args["verbose"]
    predictions,im = run(args)
    print("Predictions", predictions)