import numpy as np
# load dogs vs cats dataset, reshape and save to a new file
from os import listdir
from numpy import asarray
from numpy import save
import cv2 as cv

def sigmoid(z):
    return 1./(1.+np.exp(-z))

#this is to make binary file of the data. only needs to be run once.
if 0 :
    # dataset used : cats and dogs dataset for binary classification from kaggle
    # define location of dataset
    folder = 'train/'
    photos, labels = list(), list()
    # enumerate files in the directory
    for file in listdir(folder):
        # determine class
        output = 0.0
        if file.startswith('dog'):
            output = 1.0
        # load image
        photo = cv.imread(folder + file, cv.IMREAD_GRAYSCALE)
        photo = cv.resize(photo, (200,200)).reshape((40000))
        # store
        photos.append(photo)
        labels.append(output)
    # convert to a numpy arrays
    photos = asarray(photos)
    labels = asarray(labels)
    print(photos.shape, labels.shape)
    # save the reshaped photos
    save('dogs_vs_cats_photos.npy', photos)
    save('dogs_vs_cats_labels.npy', labels)

from numpy import load
photos = load('dogs_vs_cats_photos.npy')
labels = load('dogs_vs_cats_labels.npy')
print(photos.shape, labels.shape)

#now photos are X and labels are y
#we want to find a set of weights w and bias b that will give probability that the input image is of a dog or not.

alpha = 0.25
images_idx = np.linspace(0,24999,200, dtype=int)
X = photos.T[:,images_idx]
y = labels.T[images_idx]
m = np.shape(y)[0]
y = y.reshape((1,m))
print('X',np.shape(X))
print('y',np.shape(y))
print('m',m)
w = np.zeros((np.shape(X)[0],1))
b = np.zeros((1,m))
print("starting loop")
for itr in range(500):
    #predict the labels based on current w and b
    Z = np.dot(w.T, X) + b
    #convert into probability
    A = sigmoid(Z)
    #comparison of predicted labels and actual labels
    dZ = A - y
    #calculate derivative of cost function wrt w and b
    dw = 1./m * np.dot(X, dZ.T)
    db = 1./m * np.sum(dZ)
    #update w and b to get closer to minimum cost
    w = w - alpha * dw
    b = b - alpha * db

#now test the calculated weights and biases on all data
A = sigmoid(np.dot(w.T, X) + b) # this is prediction
A[A<0.5] = 0
A[A>=0.5] = 1
correctly_classified = y[y==A] # all such labels such that predicted label matches the true label.
print("we achieved ", 100*len(correctly_classified)/m, " percent accuracy.")