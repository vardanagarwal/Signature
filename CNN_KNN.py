# -*- coding: utf-8 -*-
"""
Created on Mon May 20 05:02:18 2019

@author: Yohan
"""
"""
at k = 4 and p = 19 with scaling
Overall 85.7
Positive 89.0
Negative 82.4

at k = 4 and p = 6 with scaling
Overall 86.7%
Positive 90.0%
Negative 83.4%

at k = 2 and p = 6 with scaling
Overall 84.7
Positive 82.2%
Negative 87.2%

at k = 4 p = 1 no scaling
Overall 84.8%
Positive 86.6
Negative 83.0%

at k = 2 and p = 19 no scaling
Overall 85.6
Positive 83.2
Negative 88.0
"""

from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras import backend as K
import cv2
import glob
import numpy as np
from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator 
from sklearn.metrics import confusion_matrix 
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.preprocessing import StandardScaler


classifier_json = open('CNN_KNN_p=6_k=4_Proper.json', 'r')
loaded_model_json = classifier_json.read()
classifier_json.close()
classifier = model_from_json(loaded_model_json)
# load weights into new model
classifier.load_weights("CNN_KNN_p=6_k=4_Proper.h5")

#initialising the classifier
final_y_pred = []
final_y_pred1 = []
final_y_predOri = []
final_y_pred1Ori = []
"""
p = 6
w = str(p)
w = w.zfill(4)
classifier = Sequential()
from keras.preprocessing.image import ImageDataGenerator    
train_datagen = ImageDataGenerator(
                rescale=1./255,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True)
    
test_datagen = ImageDataGenerator(rescale=1./255)
    
training_set = train_datagen.flow_from_directory('data\\'+ w +'_training',
                                                #save_to_dir = 'preImage/train',
                       					        target_size=(64, 64),
                       					        batch_size=1,
                       					        class_mode='categorical') #class mode is binary because it has two outputs only cat and dog
    
test_set = test_datagen.flow_from_directory('data\\'+ w +'_test',
                                            # save_to_dir = 'preImage/test',
                       					    target_size=(64, 64),
                       					    batch_size=1,
                       					    class_mode='categorical')
    
#Step 1 - Convolution 
classifier.add(Convolution2D(32, 3, 3, input_shape = (64, 64, 3), activation = 'relu')) # using tensorflow its 64,64,3 using theano it will be 3,64,64

#Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2,2)))

#Step 3 - 2nd convolutional layer
classifier.add(Convolution2D(32, 3, 3, input_shape = (64, 64, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2,2)))

#Step 3 - 3rd convolution layer
classifier.add(Convolution2D(32, 3, 3, input_shape = (64, 64, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2,2)))

#Step 3 - Flattening
classifier.add(Flatten())

#Step 4 - Full Connection
classifier.add(Dense(units = 512, activation = 'relu'))
classifier.add(Dense(units = 2, activation = 'sigmoid'))


#Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy']) # if we had more than 2 output chose categorical crossentropy as loss

# Part 2 - Fitting the CNN to the images
classifier.fit_generator(training_set,
    			 steps_per_epoch=50, #no. of images on training set
    			 epochs=20,
    			 validation_data=test_set,
    			 validation_steps=20)

"""  
for j in range(100):
    print(j)
    p = j
    w = str(p)
    w = w.zfill(4)
    images = [cv2.resize(cv2.imread(file), (64,64)) for file in glob.glob('data/' + w + '_training/positives/*png')]
    imagesNeg = [cv2.resize(cv2.imread(file), (64,64)) for file in glob.glob('data/'+w+'_training/negatives/*png')]
    imagesTest = [cv2.resize(cv2.imread(file), (64,64)) for file in glob.glob('data/'+w+'_test/positives/*png')]
    imagesTestNeg = [cv2.resize(cv2.imread(file), (64,64)) for file in glob.glob('data/'+w+'_test/negatives/*png')]
    
    images1 = np.array(images)
    image1Neg = np.array(imagesNeg)
    imageTest1 = np.array(imagesTest)
    imageTest1Neg = np.array(imagesTestNeg)
    
    get_layer_output = K.function([classifier.layers[0].input, K.learning_phase()],
                                   [classifier.layers[6].output])
    # output in test mode = 0
    layer_output_test = get_layer_output([imageTest1, 0])[0]
    layer_output_test_neg = get_layer_output([imageTest1Neg, 0])[0]
    
    
    # output in train mode = 1
    layer_output_train = get_layer_output([images, 1])[0]
    layer_output_train_neg = get_layer_output([imagesNeg, 1])[0]

#print(layer_output_train)
#print(layer_output_train.shape)
    
    
    #Extending output of intermediate data to fit into classifier
    z = np.zeros((5,1), dtype = float)
    z1 = np.zeros((20,1), dtype = float)
    one = np.ones((5,1), dtype = float)
    one1 = np.ones((20,1), dtype = float)
    layer_output_test_SVM = np.append(layer_output_test, one, axis = 1)
    layer_output_test_neg_SVM = np.append(layer_output_test_neg, z, axis = 1)
    layer_output_train_SVM = np.append(layer_output_train, one1, axis = 1)
    layer_output_train_neg_SVM = np.append(layer_output_train_neg, z1, axis = 1)
    
    x_train = np.append(layer_output_train_SVM, layer_output_train_neg_SVM, axis = 0)
    x_test = np.append(layer_output_test_SVM, layer_output_test_neg_SVM, axis = 0)
    
    y_train = x_train[:,-1]
    x_train = x_train[:,:-1]
    y_test = x_test[:,-1]
    x_test = x_test[:,:-1] 
    
    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.transform(x_test)
    
    neigh = KNeighborsClassifier(n_neighbors=4)
    neigh.fit(x_train, y_train)
    knnPred = neigh.predict(x_test)
    
    final_y_predOri.extend(y_test)
    final_y_pred1Ori.extend(y_train)
    final_y_pred.extend(knnPred)
    
cm =[]
cm = confusion_matrix(final_y_predOri,final_y_pred)
