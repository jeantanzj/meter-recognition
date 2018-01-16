
###
# Credits: https://github.com/prijip/Py-Gsvhn-DigitStruct-Reader/blob/master/digitStruct.py
# http://ufldl.stanford.edu/housenumbers/train.tar.gz
# Read the digitStruct.mat file in this training set
# Crop and save the pictures with the given bounding box
###

import h5py
import numpy as np
import argparse as ap
import os
import cv2, errno
import sys, json

#
# Bounding Box
#
class BBox:
    def __init__(self):
        self.label = ""     # Digit
        self.left = 0
        self.top = 0
        self.width = 0
        self.height = 0

class DigitStruct:
    def __init__(self):
        self.name = None    # Image file name
        self.bboxList = None # List of BBox structs

# Function for debugging
def printHDFObj(theObj, theObjName):
    isFile = isinstance(theObj, h5py.File)
    isGroup = isinstance(theObj, h5py.Group)
    isDataSet = isinstance(theObj, h5py.Dataset)
    isReference = isinstance(theObj, h5py.Reference)
    print("{}".format(theObjName))
    print("    type(): {}".format(type(theObj)))
    if isFile or isGroup or isDataSet:
        # if theObj.name != None:
        #    print "    name: {}".format(theObj.name)
        print("    id: {}".format(theObj.id))
    if isFile or isGroup:
        print("    keys: {}".format(theObj.keys()))
    if not isReference:
        print("    Len: {}".format(len(theObj)))

    if not (isFile or isGroup or isDataSet or isReference):
        print(theObj)

def readDigitStructGroup(dsFile):
    dsGroup = dsFile["digitStruct"]
    return dsGroup

#
# Reads a string from the file using its reference
#
def readString(strRef, dsFile):
    strObj = dsFile[strRef]
    str = ''.join(chr(i) for i in strObj)
    return str

#
# Reads an integer value from the file
#
def readInt(intArray, dsFile):
    intRef = intArray[0]
    isReference = isinstance(intRef, h5py.Reference)
    intVal = 0
    if isReference:
        intObj = dsFile[intRef]
        intVal = int(intObj[0])
    else: # Assuming value type
        intVal = int(intRef)
    return intVal

def yieldNextInt(intDataset, dsFile):
    for intData in intDataset:
        intVal = readInt(intData, dsFile)
        yield intVal 

def yieldNextBBox(bboxDataset, dsFile):
    for bboxArray in bboxDataset:
        bboxGroupRef = bboxArray[0]
        bboxGroup = dsFile[bboxGroupRef]
        labelDataset = bboxGroup["label"]
        leftDataset = bboxGroup["left"]
        topDataset = bboxGroup["top"]
        widthDataset = bboxGroup["width"]
        heightDataset = bboxGroup["height"]

        left = yieldNextInt(leftDataset, dsFile)
        top = yieldNextInt(topDataset, dsFile)
        width = yieldNextInt(widthDataset, dsFile)
        height = yieldNextInt(heightDataset, dsFile)

        bboxList = []

        for label in yieldNextInt(labelDataset, dsFile):
            bbox = BBox()
            bbox.label = label
            bbox.left = next(left)
            bbox.top = next(top)
            bbox.width = next(width)
            bbox.height = next(height)
            bboxList.append(bbox)

        yield bboxList

def yieldNextFileName(nameDataset, dsFile):
    for nameArray in nameDataset:
        nameRef = nameArray[0]
        name = readString(nameRef, dsFile)
        yield name

# dsFile = h5py.File('../data/gsvhn/train/digitStruct.mat', 'r')
def yieldNextDigitStruct(dsFileName):
    dsFile = h5py.File(dsFileName, 'r')
    dsGroup = readDigitStructGroup(dsFile)
    nameDataset = dsGroup["name"]
    bboxDataset = dsGroup["bbox"]

    bboxListIter = yieldNextBBox(bboxDataset, dsFile)
    for name in yieldNextFileName(nameDataset, dsFile):
        bboxList = next(bboxListIter)
        obj = DigitStruct()
        obj.name = name
        obj.bboxList = bboxList
        yield obj

def showImage(img, title):

    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.imshow(title, img)
    cv2.waitKey(0)

def run(dsFileName, outFolder):
    errors=[]
    #testCounter = 0
    par=os.path.abspath(os.path.join(dsFileName,os.pardir))
    for dsObj in yieldNextDigitStruct(dsFileName):
        #testCounter += 1
        im = cv2.imread(os.path.join(par,dsObj.name))
        im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        m = np.median(im_gray)
        #showImage(im_gray,"im_gray")
        # for th in range(0,200,20):
        #     print(th)
        #(_, im_bw) = cv2.threshold(im_gray, (1-0.33)*m , 255, cv2.THRESH_BINARY_INV)
        #showImage(im_bw,"")
        output = im_gray
        print(dsObj.name)
        cnt = 0
        for bbox in dsObj.bboxList:
            try:
                cnt+=1
                label = 0 if bbox.label == 10 else bbox.label
                print("    {}:{},{},{},{}".format(
                    label, bbox.left, bbox.top, bbox.width, bbox.height))
                # leng = int(bbox.width * 1.6) #tan 45 deg = height/length
                # pt1 = max( int(bbox.top + bbox.height // 2 - leng // 2),0)
                # pt2 = max( int(bbox.left + bbox.width // 2 - leng // 2) , 0)
                # draw.rectangle((bbox.left, bbox.top, bbox.left+leng , bbox.top+leng) , fill=None) 
                
                
                #cv2.rectangle(output, (bbox.left, bbox.top), (bbox.left + bbox.width, bbox.top + bbox.height), (128, 255, 0), 2) 
                roi = output[bbox.top:bbox.top+bbox.height,bbox.left:bbox.left+bbox.width]
                roi = cv2.resize(roi, (28, 28)) 
                
                vpath=os.path.join(outFolder,str(label)+"/"+dsObj.name+"-"+str(cnt)+".png")
                cv2.imwrite(vpath, roi)
            except Exception as e:
                errors.append(json.dumps({"err": str(e), "name": dsObj.name, "label": bbox.label }))
        
            #showImage(roi, "Item")
            #v = im_bw.crop((bbox.left, bbox.top, bbox.width+bbox.left, bbox.height+bbox.top))
            
            #v.save(vpath) 
        #if testCounter >= 5:
         #   break
    if len(errors) > 0:
        print( len(errors) , "occurred")
        with open("errors.json",'w') as errfile:
            errfile.write(json.dumps(errors))      

if __name__ == "__main__":

    parser = ap.ArgumentParser()
    parser.add_argument("-i", "--digitStruct", help="Path to digitStruct", required="True")
    parser.add_argument("-o", "--outFolder", help="out Folder")
    args = vars(parser.parse_args())
    print("Args:", args)

    if args["outFolder"] is None:
        upperpar=os.path.abspath(os.path.join(args["digitStruct"],os.pardir))
        outFolder = os.path.join(upperpar,"data/")
    else:
        outFolder = os.path.abspath(args["outFolder"])
    
    try:
        os.mkdir(outFolder)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
        pass
    for i in range(0,10):
        try:
            os.mkdir(os.path.join(outFolder,str(i)))
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise
            pass        
    run(args["digitStruct"], outFolder)
