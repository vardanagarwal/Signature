# -*- coding: utf-8 -*-
"""
Created on Fri May 31 09:51:03 2019

@author: hp
"""

from imutils import paths
import cv2
for folders in range(0,100):
    if folders < 10:
        training = "data\\000" + str(folders) + "_training"
        test = "data\\000" + str(folders) + "_test"
    else:
        training = "data\\00" + str(folders) + "_training"
        test = "data\\00" + str(folders) + "_test"
    # initialize the data matrix and labels
    print ("[INFO] extracting features...", folders)    
    # loop over the image paths in the training set
    for imagePath in paths.list_images(training):
        image = cv2.imread(imagePath)
        image = image[34:620, 46:820, :]
        cv2.imwrite(imagePath, image)
    for imagePath in paths.list_images(test):
        image = cv2.imread(imagePath)
        image = image[34:620, 46:820, :]
        cv2.imwrite(imagePath, image)