# -*- coding: utf-8 -*-
"""
Created on Fri May 17 01:19:35 2019

@author: hp


ACCURACY OBTAINED:
    negative accuracy =  0.8639999999999993
    positive accuracy =  0.9079999999999996

"""


# import the necessary packages
#from skimage import exposure
from skimage import feature
from imutils import paths
import pandas as pd
import cv2


result = [[],[]] 
result_as_01 = []
result_as_percent = []
original = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
for folders in range(0,100):
    if folders < 10:
        training = "data\\000" + str(folders) + "_training"
        test = "data\\000" + str(folders) + "_test"
    else:
        training = "data\\00" + str(folders) + "_training"
        test = "data\\00" + str(folders) + "_test"
    # initialize the data matrix and labels
    print ("[INFO] extracting features...", folders)
    data = []
    labels = []
    
    # loop over the image paths in the training set
    for imagePath in paths.list_images(training):
    	# extract the make of the car
        make = imagePath.split("\\")[-2]
     
    	# load the image, convert it to grayscale, and detect edges
        image = cv2.imread(imagePath)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#    	edged = imutils.auto_canny(gray)
#     
#
#    	cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
#    		cv2.CHAIN_APPROX_SIMPLE)
#    	cnts = cnts[0] if imutils.is_cv2() else cnts[1]
#    	c = max(cnts, key=cv2.contourArea)
#     
#    	# extract the logo of the car and resize it to a canonical width
#    	# and height
#    	(x, y, w, h) = cv2.boundingRect(c)
#    	logo = gray[y:y + h, x:x + w]
        gray = cv2.resize(gray, (128, 128))
     
    	# extract Histogram of Oriented Gradients from the logo
        H = feature.hog(gray, orientations=9, pixels_per_cell=(8, 8),
    		cells_per_block=(2, 2), transform_sqrt=True, block_norm="L1")
     
    	# update the data and labels
        data.append(H)
        if make == 'positives':
            labels.append(1)
        else:
            labels.append(0)
    data = pd.DataFrame(data)
    labels = pd.DataFrame(labels)
    # "train" the ann
    print("[INFO] training classifier...")
    #Splitting the dataset
    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size = 0.25, random_state = 0)
    from keras.models import Sequential
    from keras.layers import Dense
    
    #Initializing the ANN
    classifier = Sequential()
    
    #Adding the input layer and the first hidden layer
    classifier.add(Dense(units = 32, kernel_initializer = 'uniform', activation = 'relu', input_dim = 8100))
    
    #Adding the second hidden layer
    classifier.add(Dense(units = 32, kernel_initializer = 'uniform', activation = 'relu'))
    
    #Adding the output layer
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    
    #compiling the ann
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    
    #Fitting the ANN to the Training Set
    classifier.fit(x_train, y_train, batch_size = 4, epochs = 30)
    print("[INFO] evaluating...")
    # loop over the test dataset
    pred = []
    for (i, imagePath) in enumerate(paths.list_images(test)):
    	# load the test image, convert it to grayscale, and resize it to
    	# the canonical size
        image = cv2.imread(imagePath)
            
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        logo = cv2.resize(gray, (128, 128))
     
    	# extract Histogram of Oriented Gradients from the test image and
    	# predict the make of the car
        (H, hogImage) = feature.hog(logo, orientations=9, pixels_per_cell=(8, 8),
    		cells_per_block=(2, 2), transform_sqrt=True, block_norm="L1", visualize=True)
        pred.append(classifier.predict(H.reshape(1, -1))[0] > 0.5)
        result_as_01.append(int(classifier.predict(H.reshape(1, -1))[0] > 0.5))
        result_as_percent.append(classifier.predict(H.reshape(1, -1))[0])
    	# visualize the HOG image
#        hogImage = exposure.rescale_intensity(hogImage, out_range=(0, 255))
#        hogImage = hogImage.astype("uint8")
#        cv2.imshow("HOG Image #{}".format(i + 1), hogImage)
        
    	# draw the prediction on the test image and display it
#        cv2.putText(image, pred[i].title(), (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
#    		(0, 255, 0), 3)
#        cv2.imshow("Test Image #{}".format(i + 1), image)
#        cv2.waitKey(0)
#    cv2.destroyAllWindows()
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
    print("[INFO] negatives accuracy = ", neg_result, " and positives accuray = ", pos_result)
negs = sum(result[0])/100
poss = sum(result[1])/100
print("negative accuracy = ", negs)
print("positive accuracy = ", poss)