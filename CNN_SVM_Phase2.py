# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 21:30:36 2019

@author: Yohan
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Feb  2 01:32:07 2019

@author: hp
"""
# Part 1 - Building the CNN
from keras.models import Sequential
from keras.layers import Convolution2D # 2D is for images 3D is for videos time is the third dimension
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras import backend as K

#initialising the classifier
classifier = Sequential()
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('data\\0000_training',
                                                #save_to_dir = 'preImage/train',
                    					        target_size=(64, 64),
                    					        batch_size=1,
                    					        class_mode='categorical') #class mode is binary because it has two outputs only cat and dog

test_set = test_datagen.flow_from_directory('data\\0000_test',
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
from keras.models import Model

#get_3rd_layer_output = K.function([classifier.layers[0].input, K.learning_phase()],
#                                  [classifier.layers[3].output])
#
## output in test mode = 0
## output in test mode = 0
#layer_output = get_3rd_layer_output([test_set, 0])[0]
#
## output in train mode = 1
#layer_output = get_3rd_layer_output([training_set, 1])[0]
#Step 4 - Full Connection
classifier.add(Dense(units = 512, activation = 'relu'))
classifier.add(Dense(units = 2, activation = 'sigmoid'))


#Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy']) # if we had more than 2 output chose categorical crossentropy as loss

#classifier = VGG16()
# Part 2 - Fitting the CNN to the images




classifier.fit_generator(training_set,
			 steps_per_epoch=50, #no. of images on training set
			 epochs=20,
			 validation_data=test_set,
			 validation_steps=20)
"""
from PIL import Image
import requests
from io import BytesIO

#response = requests.get('http://thehappypet.co/wp-content/uploads/2015/09/funny-dog-hair-cuts-1000x600.jpg')
test_image = Image.open("new/0001/0001f01.png")
test_image = test_image.resize(size = (64, 64))
import numpy as np
from keras.preprocessing import image
#test_image = image.load_img('https://www.catster.com/wp-content/uploads/2018/07/Savannah-cat-long-body-shot.jpg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
training_set.class_indices
"""
"""if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'
"""
"""
model_json = classifier.to_json()
with open("model.json","w") as json_file:
    json_file.write(model_json)
    
# serialize weights to HDF5
classifier.sample_weights("model.h5")
"""

import cv2
import glob
from keras import backend as K
import numpy as np
images = [cv2.resize(cv2.imread(file), (64,64)) for file in glob.glob('data/0000_training/positives/*png')]
imagesNeg = [cv2.resize(cv2.imread(file), (64,64)) for file in glob.glob('data/0000_training/negatives/*png')]
imagesTest = [cv2.resize(cv2.imread(file), (64,64)) for file in glob.glob('data/0000_test/positives/*png')]
imagesTestNeg = [cv2.resize(cv2.imread(file), (64,64)) for file in glob.glob('data/0000_test/negatives/*png')]

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



print(layer_output_train)
print(layer_output_train.shape)

"""
newModel = Model(classifier.inputs,classifier.layers[6].output)
newModel.summary()
wer = newModel.predict(images1, verbose = 1)
werNeg = newModel.predict(image1Neg, verbose = 1)
werTest = newModel.predict(imageTest1, verbose = 1)
werTestNeg = newModel.predict(imageTest1Neg, verbose = 1)

qwe2 = newModel.predict_generator(training_set, verbose = 1,steps = 1)
asd = newModel.predict_generator(test_set, verbose = 1)
"""


#Extending output of intermediate data to fit into svm
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
#x_train_shuffle = np.random.shuffle(x_train)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

from sklearn.svm import SVC
classifier1 = SVC(kernel = 'linear', random_state = 0)
classifier1.fit(x_train, y_train)

y_pred = classifier1.predict(x_test)

