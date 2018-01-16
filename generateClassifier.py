#!/usr/bin/python

# Import the modules
from sklearn.externals import joblib
from sklearn import datasets
from skimage.feature import hog
from sklearn.svm import LinearSVC
from sklearn import preprocessing
import numpy as np
from collections import Counter
import gzip
IMAGE_SIZE = 28
# Load the dataset
# dataset = datasets.fetch_mldata("MNIST Original")
#outputFile='digits_cls_mnist.pkl'
outputFile='digits_cls_svhn_nobw.pkl'
total = 73228
limit = 10000
imagesFile = 'generateYourOwnData/train-nobw-images-idx3-ubyte-'+str(total)+'.gz'
labelsFile = 'generateYourOwnData/train-nobw-labels-idx1-ubyte-'+str(total)+'.gz'


def extract_data(filename, num_images):
  """Extract the images into a 4D tensor [image index, y, x, channels].
  Values are rescaled from [0, 255] down to [-0.5, 0.5].
  """
  print('Extracting', filename)
  with gzip.open(filename) as bytestream:
    bytestream.read(16)
    buf = bytestream.read(IMAGE_SIZE * IMAGE_SIZE * num_images)
    data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
    #data = (data - (PIXEL_DEPTH / 2.0)) / PIXEL_DEPTH
    data = data.reshape(num_images, IMAGE_SIZE, IMAGE_SIZE, 1)
    return data


def extract_labels(filename, num_images):
  """Extract the labels into a vector of int64 label IDs."""
  print('Extracting', filename)
  with gzip.open(filename) as bytestream:
    bytestream.read(8)
    buf = bytestream.read(1 * num_images)
    labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
  return labels


# Extract the features and labels
features = extract_data(imagesFile,total)
labels = extract_labels(labelsFile,total)

list_hog_fd = []
list_labels = []
for i in range(0,10):
  a =  np.where(labels == i)[0]
  size = min(len(a), limit)
  #sizes.append(size)
  chosen = np.random.choice(a, size=size, replace=False)
  for j in chosen:
    fd = hog(features[j].reshape((28, 28)), orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualise=False)
    list_hog_fd.append(fd)
    list_labels.append(labels[j])
#features = np.array(dataset.data, 'int16') 
#labels = np.array(dataset.target, 'int')
# Extract the hog features

# for feature in features:
#     fd = hog(feature.reshape((28, 28)), orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualise=False)
#     list_hog_fd.append(fd)
hog_features = np.array(list_hog_fd, 'float64')

# Normalize the features
pp = preprocessing.StandardScaler().fit(hog_features)
hog_features = pp.transform(hog_features)
labels = list_labels
#labels = np.repeat(np.arange(0,10), sizes, axis=0)
print "Count of digits in dataset", Counter(labels)

# Create an linear SVM object
clf = LinearSVC()

# Perform the training
clf.fit(hog_features, labels)

# Save the classifier
joblib.dump((clf, pp), outputFile, compress=3)
