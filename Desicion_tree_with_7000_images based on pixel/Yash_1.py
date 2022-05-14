#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 26 12:09:34 2022

@author: param-mu-scom
"""

import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
#from sklearn.metrics import pri_score
from sklearn.metrics import precision_recall_fscore_support

path = "/home/param-mu-scom/ICT_Yash/YASHDATA/train_images/train.csv"
path2= "/home/param-mu-scom/ICT_Yash/YASHDATA/train_images"
dataset = pd.read_csv(path)


print(dataset)

data=dataset.values[:7095,0:2]#7000 image for train
print(data.shape)
print(data[1][0])
import cv2
import os
image=[]
classes=[]


for i in range(7094):   
  name=data[i][0]
  join_path=os.path.join(path2,name)  
  img=cv2.imread(join_path)#read 5000 setof image
  img2=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
  image.append(img2)#append all the image in imge
  classes.append(data[i][1])#append all the class in classes
  #print(name)
  #print(classes)
print(image)
print(classes)
image=np.array(image)
classes=np.array(classes)
#px.imshow(img2,binary_string=True)

print(classes.shape)

from sklearn.model_selection import train_test_split
x_trainset, x_testset, y_trainset, y_testset = train_test_split(image,classes, test_size=0.3, random_state=3)



print(type(x_trainset))


#traing set
print('Shape of X training set {}'.format(x_trainset.shape),'&',' Size of Y training set {}'.format(y_trainset.shape))

#test set
print('Shape of X training set {}'.format(x_testset.shape),'&',' Size of Y training set {}'.format(y_testset.shape))

from sklearn.tree import DecisionTreeClassifier
imageTree = DecisionTreeClassifier(criterion="entropy", max_depth = 4)
imageTree # it shows the default parameters

nsamples, nx, ny = x_trainset.shape
new_x_train = x_trainset.reshape((nsamples,nx*ny))

nsamples1, nx1, ny1 = x_testset.shape
new_x_test = x_testset.reshape((nsamples1,nx1*ny1))

print(new_x_test.shape)
print(new_x_train.shape)



imageTree.fit(new_x_train,y_trainset)
pred_class = imageTree.predict(new_x_test)

print(y_testset)
print(pred_class.shape)

#depth at 4 ->0.69
#depth at 10 ->0.63
#depth at 3 ->0.65
#depth at 5 ->0.68
#depth at 6 ->0.66
from sklearn import metrics
import matplotlib.pyplot as plt
print("DecisionTrees's Accuracy: ", metrics.accuracy_score(y_testset, pred_class))

print("DecisionTrees's f1 score: ", f1_score(y_testset, pred_class,average=None))
print("DecisionTrees's recall: ", recall_score(y_testset, pred_class,average=None))

print("DecisionTrees's recall: ", precision_recall_fscore_support(y_testset, pred_class,average='micro'))
#print(new_x_test.shape)





train_accuracy = list()
test_accuracy = list()
max_depth = list()

for i in range(1,5) :
    imageTree = DecisionTreeClassifier(criterion="entropy", max_depth = i)
    
    print("for depth",i)
    imageTree.fit(new_x_train,y_trainset)
    pred_class = imageTree.predict(new_x_train)
    print("DecisionTrees's Training Accuracy: ", metrics.accuracy_score(y_trainset, pred_class))
    print("DecisionTrees's Training f1 score: ", f1_score(y_trainset, pred_class,average='weighted'))
    print("DecisionTrees's  Training recall: ", recall_score(y_trainset, pred_class,average='weighted'))
    print("DecisionTrees's  Training precision: ", precision_score(y_trainset, pred_class,average='weighted'))

    print("DecisionTrees's Training All: ", precision_recall_fscore_support(y_trainset, pred_class,average='micro'))
    train_accuracy.append(metrics.accuracy_score(y_trainset, pred_class))
    
    
    #imageTree.fit(new_x_train,y_trainset)
    pred_test_class = imageTree.predict(new_x_test)
    print("DecisionTrees's Testing Accuracy: ", metrics.accuracy_score(y_testset, pred_test_class))
    print("DecisionTrees's Testing f1 score: ", f1_score(y_testset, pred_test_class,average='weighted'))
    print("DecisionTrees's  Testing recall: ", recall_score(y_testset, pred_test_class,average='weighted')) 
    print("DecisionTrees's  Training precision: ", precision_score(y_testset, pred_test_class,average='weighted'))

    print("DecisionTrees's Testing All: ", precision_recall_fscore_support(y_testset, pred_test_class,average='micro'))
    test_accuracy.append(metrics.accuracy_score(y_testset, pred_test_class))    
    max_depth.append(i)


print(train_accuracy)
print(test_accuracy)
print(max(test_accuracy))
