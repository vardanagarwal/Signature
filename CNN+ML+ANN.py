# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 22:06:22 2019

@author: Yohan
"""

"""
With ANN 96.5% in training and 96% in testing

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
from keras.applications.vgg16 import VGG16
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xlwt import Workbook 
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.layers import Dense

classifier = VGG16()

final_y_pred = []
final_y_pred1 = []
final_y_predOri = []
final_y_pred1Ori = []
knnTrainPredAll = []
knnTestPredAll = []
svmTrainPredAll = []
svmTestPredAll = []
randomForestTrainPredAll = []
randomForestTestPredAll = []
extraTreesTrainPredAll = []
extraTreesTestPredAll = []

class Excel:
    def __init__(self, wb, sheet):
        self.wb = wb
        self.i = 1
        self.sheet = sheet
        sheet.write(0, 0, 'KNN')
        sheet.write(0, 1,'Extra')
        sheet.write(0, 2,'SVM')
        sheet.write(0, 3, 'Random')
        sheet.write(0, 4, 'Actual')
    
    def AddToExcel(self, listOut, col):
        #for i in range(len(listOut)-1):
        [self.sheet.write(i+1, col, listOut[i]) for i in range(len(listOut))]
        
class listData:
    def listToExcel(self, a, b, c, d, e, f, g, h):
        wb1 = Workbook()
        sheetWb1 = wb1.add_sheet('Sheet1', cell_overwrite_ok='True')
        
        excel1 = Excel(wb1, sheetWb1)
        excel1.AddToExcel(a, 0)
        
        excel1.AddToExcel(b, 1)
        
        excel1.AddToExcel(c, 2)
        
        excel1.AddToExcel(d, 3)
        
        excel1.AddToExcel(final_y_pred1Ori, 4)
        
        wb1.save('TrainPred1.xls')
        
        wb2 = Workbook()
        sheetWb2 = wb2.add_sheet('Sheet1', cell_overwrite_ok='True')
        
        excel2 = Excel(wb2, sheetWb2)
        excel2.AddToExcel(e, 0)
        
        excel2.AddToExcel(f, 1)
        
        excel2.AddToExcel(g, 2)
        
        excel2.AddToExcel(h, 3)
        
        excel2.AddToExcel(final_y_predOri, 4)
        
        wb2.save('TestPred1.xls')
        
        return True
        

for j in range(100):
    print(j)
    p = j
    w = str(p)
    w = w.zfill(4)
    images = [cv2.resize(cv2.imread(file), (224,224)) for file in glob.glob('data/' + w + '_training/positives/*png')]
    imagesNeg = [cv2.resize(cv2.imread(file), (224,224)) for file in glob.glob('data/'+w+'_training/negatives/*png')]
    imagesTest = [cv2.resize(cv2.imread(file), (224,224)) for file in glob.glob('data/'+w+'_test/positives/*png')]
    imagesTestNeg = [cv2.resize(cv2.imread(file), (224,224)) for file in glob.glob('data/'+w+'_test/negatives/*png')]
    
    images1 = np.array(images)
    image1Neg = np.array(imagesNeg)
    imageTest1 = np.array(imagesTest)
    imageTest1Neg = np.array(imagesTestNeg)
    
    get_layer_output = K.function([classifier.layers[0].input, K.learning_phase()],
                                   [classifier.layers[19].output])
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
    """
    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.transform(x_test)
    """
    
    neigh = KNeighborsClassifier(n_neighbors=2)
    neigh.fit(x_train, y_train)
    knnPred = neigh.predict(x_test)
    knnTrainPred = neigh.predict(x_train)
    knnTrainPredAll.extend(knnTrainPred)
    knnTestPredAll.extend(knnPred)
    
    trees = ExtraTreesClassifier(n_estimators = 1400, max_features = 'log2')
    trees.fit(x_train, y_train)
    extraTreesPred = trees.predict(x_test)
    extraTreesTrainPred = trees.predict(x_train)
    extraTreesTrainPredAll.extend(extraTreesTrainPred)
    extraTreesTestPredAll.extend(extraTreesPred)
    
    classifier1 = SVC(kernel = 'linear', random_state = 0)
    classifier1.fit(x_train, y_train)
    svmPred = classifier1.predict(x_test)
    svmTrainPred = classifier1.predict(x_train)
    svmTrainPredAll.extend(svmTrainPred)
    svmTestPredAll.extend(svmPred)
    
    
    forest = RandomForestClassifier(n_estimators=300)
    forest.fit(x_train, y_train)
    randomForestPred = forest.predict(x_test)
    randomForestTrainPred = forest.predict(x_train)
    randomForestTrainPredAll.extend(randomForestTrainPred)
    randomForestTestPredAll.extend(randomForestPred)
    
    final_y_predOri.extend(y_test)
    final_y_pred1Ori.extend(y_train)
    #final_y_pred.extend(knnPred)
  
""" 
cm =[]
cm = confusion_matrix(final_y_predOri,final_y_pred)
"""

cm1 = []
cm1 = confusion_matrix(final_y_pred1Ori, knnTrainPredAll)

cm2 = []
cm2 = confusion_matrix(final_y_pred1Ori, svmTrainPredAll)

cm3 = []
cm3 = confusion_matrix(final_y_pred1Ori, extraTreesTrainPredAll)

cm4 = []
cm4 = confusion_matrix(final_y_pred1Ori, randomForestTrainPredAll)

cm5 = []
cm5 = confusion_matrix(final_y_predOri, knnTestPredAll)

cm6 = []
cm6 = confusion_matrix(final_y_predOri, extraTreesTestPredAll)

cm7 = []
cm7 = confusion_matrix(final_y_predOri, svmTestPredAll)

cm8 = []
cm8 = confusion_matrix(final_y_predOri, randomForestTestPredAll)

listData1 = listData()
listData1.listToExcel(knnTrainPredAll, extraTreesTrainPredAll, svmTrainPredAll,
                      randomForestTrainPredAll, knnTestPredAll,
                      extraTreesTestPredAll, svmTestPredAll,
                      randomForestTestPredAll)

dataset = pd.read_excel('TestPred.xls')
X = dataset.iloc[:, 0:4]
Y = dataset.iloc[:,-1]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

annClassifier = Sequential()

annClassifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 4))

# Adding the second hidden layer
annClassifier.add(Dense(units = 12, kernel_initializer = 'uniform', activation = 'relu'))

#annClassifier.add(Dense(units = 12, kernel_initializer = 'uniform', activation = 'relu'))

annClassifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))

# Adding the output layer
annClassifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# compiling the ANN
annClassifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the training set
annClassifier.fit(X_train, Y_train, batch_size = 20, epochs = 100)
# Part 3 - Making the predictions and evaluating the model

# Fitting the classifier to the Training set


# Predicting the Test set results
y_predAnn = annClassifier.predict(X_test)
y_predAnn = [1 if i > 0.5 else 0 for i in y_predAnn]
Y_test = list(Y_test)
cm9 = []
cm9 = confusion_matrix(Y_test,y_predAnn)

y_predTrainAnn = annClassifier.predict(X_train)
y_predTrainAnn = [1 if i > 0.5 else 0 for i in y_predTrainAnn]
Y_train = list(Y_train)
cm10 = []
cm10 = confusion_matrix(Y_train,y_predTrainAnn)

#pickleData = pd.DataFrame({"KNN" : knnTestPredAll, "Extra Trees": extraTreesTestPredAll, "SVM" : svmTestPredAll, "Random Forest" : randomForestTestPredAll})
#pickleData.to_pickle("./Test.pkl")







