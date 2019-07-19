# -*- coding: utf-8 -*-
"""
Created on Sat Jun  1 20:41:35 2019

@author: hp
"""

# -*- coding: utf-8 -*-
"""
Created on Fri May 31 09:25:00 2019

@author: hp

ACCURACY OBTAINED:
    negative accuracy =  0.9399999999999996
    positive accuracy =  0.9619999999999997
"""


import numpy as np
from imutils import paths
import pandas as pd

from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input

#model_vgg.summary()

result = [[],[]] 
original = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
def first_nonzero(arr, axis, invalid_val=-1):
    mask = arr!=0
    return np.where(mask.any(axis=axis), mask.argmax(axis=axis), invalid_val)
def last_nonzero(arr, axis, invalid_val=-1):
    mask = arr!=0
    val = arr.shape[axis] - np.flip(mask, axis=axis).argmax(axis=axis) - 1
    return np.where(mask.any(axis=axis), val, invalid_val)
for folders in range(0,100):
    if folders < 10:
        training = "data\\000" + str(folders) + "_training"
        test = "data\\000" + str(folders) + "_test"
    else:
        training = "data\\00" + str(folders) + "_training"
        test = "data\\00" + str(folders) + "_test"
    # initialize the data matrix and labels
    print ("[INFO] extracting features...", folders)
    vgg16_feature_list = []
    labels = []
    model_vgg = VGG16(weights = 'imagenet', include_top = False)
    # loop over the image paths in the training set
    for imagePath in paths.list_images(training):
    	# extract the make of the car
        make = imagePath.split("\\")[-2]
    	# load the image, convert it to grayscale, and detect edges
        img = image.load_img(imagePath, target_size = (224, 224))
        img_data = image.img_to_array(img)
        img_data = np.expand_dims(img_data, axis = 0)
        img_data = preprocess_input(img_data)

        vgg16_feature = model_vgg.predict(img_data)
        vgg16_feature_np = np.array(vgg16_feature)
        vgg16_feature_list.append(vgg16_feature_np.flatten())
        if make == 'positives':
            labels.append(1)
        else:
            labels.append(0)
    vgg16_feature_list_np = np.array(vgg16_feature_list)
    print("[INFO] training classifier...")
    
    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(vgg16_feature_list_np, labels, test_size = 0.25, random_state = 0)
    from keras.models import Sequential
    from keras.layers import Dense
    
    #Initializing the ANN
    classifier = Sequential()
    
    #Adding the input layer and the first hidden layer
    classifier.add(Dense(units = 32, kernel_initializer = 'uniform', activation = 'relu', input_dim = 25088))
    
    #Adding the second hidden layer
    classifier.add(Dense(units = 32, kernel_initializer = 'uniform', activation = 'relu'))
    
    #Adding the output layer
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    
    #compiling the ann
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    
    #Fitting the ANN to the Training Set
    classifier.fit(x_train, y_train, batch_size = 4, epochs = 12)
    print("[INFO] evaluating...")    
    
    pred = []
    for (i, imagePath) in enumerate(paths.list_images(test)):
        img = image.load_img(imagePath, target_size = (224, 224))
        img_data = image.img_to_array(img)
        img_data = np.expand_dims(img_data, axis = 0)
        img_data = preprocess_input(img_data)
        
        vgg16_feature = model_vgg.predict(img_data)
        vgg16_feature_np = np.array(vgg16_feature)
        pred.append(classifier.predict(vgg16_feature_np.reshape(1, -1))[0] > 0.5)
        
    neg_result = 0
    pos_result = 0
    for i in range(0,5):
        if pred[i] == original[i]:
            neg_result = neg_result + 1
    for i in range(5,10):
        if pred[i] == original[i]:
            pos_result = pos_result + 1
    neg_result = neg_result/5
    pos_result = pos_result/5        
    result[0].append(neg_result)
    result[1].append(pos_result)
    print(neg_result, pos_result)

negs = sum(result[0])/100
poss = sum(result[1])/100
print("negative accuracy = ", negs)
print("positive accuracy = ", poss)
        