# -*- coding: utf-8 -*-
"""
Created on Mon May 20 12:12:50 2019

@author: hp
"""
#train classifier
# serialize model to JSON
model_json = classifier.to_json()
with open("model1.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
classifier.save_weights("model1.h5")
print("Saved model to disk")

# load json and create model
from keras.models import model_from_json

json_file = open('model1.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model1.h5")
#print("Loaded model from disk")