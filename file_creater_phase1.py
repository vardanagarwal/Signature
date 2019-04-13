# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 00:07:00 2019

@author: hp
"""

import os
import re

os.makedirs("data_phase1\\training")
os.makedirs("data_phase1\\test")

name = "data_phase1"
folder = os.listdir(name)
for file in folder:
    src = name+"\\"+file
    dst = name+"\\training\\"+file
    if re.search('0', file) != None:
        os.rename(src, dst)
        
name = "data_phase1\\training"
folder = os.listdir(name)
for files in folder:
    file = os.listdir(name+"\\"+files)
    for images in file:
        if re.search('f', images) != None:
            os.remove(name+"\\"+files+"\\"+images)
        
name = "data_phase1\\training"
folder = os.listdir(name)
for files in folder:
    file = os.listdir(name+"\\"+files)
    os.makedirs("data_phase1\\test\\"+str(files))
    for images in file:
        if re.search('3', images) or re.search('8', images) != None:
            os.rename(name+"\\"+files+"\\"+images, "data_phase1\\test\\" + files + "\\" + images)