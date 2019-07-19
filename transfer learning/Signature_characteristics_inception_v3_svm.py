# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 17:39:26 2019

@author: hp
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 13:45:23 2019

@author: hp


"""

import numpy as np
from imutils import paths

from keras.preprocessing import image
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input

from sklearn.svm import SVC

#model_vgg.summary()

result = [[],[]] 
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
    model_inception_v3 = InceptionV3(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
    # loop over the image paths in the training set
    for imagePath in paths.list_images(training):
    	# extract the make of the car
        make = imagePath.split("\\")[-2]
    	# load the image, convert it to grayscale, and detect edges
        img = image.load_img(imagePath, target_size = (224, 224))
        img_data = image.img_to_array(img)
        img_data = np.expand_dims(img_data, axis = 0)
        img_data = preprocess_input(img_data)

        vgg16_feature = model_inception_v3.predict(img_data)
        vgg16_feature_np = np.array(vgg16_feature)
        vgg16_feature_list.append(vgg16_feature_np.flatten())
        labels.append(make)
    vgg16_feature_list_np = np.array(vgg16_feature_list)
    print("[INFO] training classifier...")
    model = SVC(kernel = 'linear', random_state = 0)
    model.fit(vgg16_feature_list_np, labels)
    print("[INFO] evaluating...")
    
    pred = []
    original = []
    for (i, imagePath) in enumerate(paths.list_images(test)):
        original.append(imagePath.split("\\")[-2])        
        img = image.load_img(imagePath, target_size = (299, 299))
        img_data = image.img_to_array(img)
        img_data = np.expand_dims(img_data, axis = 0)
        img_data = preprocess_input(img_data)
        
        vgg16_feature = model_inception_v3.predict(img_data)
        vgg16_feature_np = np.array(vgg16_feature)
        pred.append(str(model.predict(vgg16_feature_np.reshape(1, -1))[0]))
        
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
        
