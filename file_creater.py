# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 01:24:43 2019

@author: hp
"""

import os

for i in range(0,100):
    if i<10:
        positives = "data\\000" + str(i) + "\positives"
        negatives = "data\\000" + str(i) + "\\negatives"
    else:
        positives = "data\\00" + str(i) + "\positives"
        negatives = "data\\00" + str(i) + "\\negatives"        
    os.makedirs(positives)
    os.makedirs(negatives)

import re

for i in range(0,100):
    if i<10:
        name = "data\\000" + str(i)
    else:
        name = "data\\00" + str(i)
    
    files = os.listdir(name)
    for file in files:
        if re.search("png",file) != None:
            if re.search("v",file) != None:
                src = name + "\\" + file
                dst = name + "\\positives\\" + file
                os.rename(src, dst)
            else:
                src = name + "\\" + file
                dst = name + "\\negatives\\" + file
                os.rename(src, dst)
                
folders = os.listdir("data")
for folder in folders:
    src = "data\\" + folder
    dst = "data\\" + folder + "_training"
    new = "data\\" + folder + "_test"
    os.rename(src, dst)
    os.makedirs(new)
    
for i in range(0,100):
    if i<10:
        positives = "data\\000" + str(i) + "_test\positives"
        negatives = "data\\000" + str(i) + "_test\\negatives"
    else:
        positives = "data\\00" + str(i) + "_test\positives"
        negatives = "data\\00" + str(i) + "_test\\negatives"        
    os.makedirs(positives)
    os.makedirs(negatives)
    

for i in range(0, 100):
    if i<10:
        p_src = "data\\000" + str(i) + "_training\positives"
        n_src = "data\\000" + str(i) + "_training\\negatives"
        p_dst = "data\\000" + str(i) + "_test\positives"
        n_dst = "data\\000" + str(i) + "_test\\negatives"
    else:
        p_src = "data\\00" + str(i) + "_training\positives"
        n_src = "data\\00" + str(i) + "_training\\negatives"
        p_dst = "data\\00" + str(i) + "_test\positives"
        n_dst = "data\\00" + str(i) + "_test\\negatives"
    p_folders = os.listdir(p_src)
    n_folders = os.listdir(n_src)
    for image in p_folders:
        p_src1 = p_src + "\\" + image
        if re.search('4', re.split('v',image)[1]) or re.search('9', re.split('v',image)[1]) != None:
            p_dst1 = p_dst + "\\" + image
            os.rename(p_src1, p_dst1)
    for image in n_folders:
        n_src1 = n_src + "\\" + image
        if re.search('4', re.split('f',image)[1]) or re.search('9', re.split('f',image)[1]) != None:
            n_dst1 = n_dst + "\\" + image
            os.rename(n_src1, n_dst1)    
            

                