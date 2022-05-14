import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import precision_recall_fscore_support
import cv2
import os
from sklearn import metrics
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score

#path for train set image
path = "/home/param-mu-scom/ICT_Yash/YASHDATA/train_images/train.csv"
path2= "/home/param-mu-scom/ICT_Yash/YASHDATA/train_images"

#read csv file
dataset = pd.read_csv(path)
print(dataset)
data=dataset.values[:7095,0:2]#7000 image for train


image=[]
classes=[]

#taking image from csv file
for i in range(7095):   
  name=data[i][0]
  join_path=os.path.join(path2,name)  
  img=cv2.imread(join_path)#read 5000 setof image
  img2=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
  image.append(img2)#append all the image in imge
  classes.append(data[i][1])#append all the class in classes
  


#converitng into numpy
image=np.array(image)
classes=np.array(classes)




#split train and test
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(image, classes, test_size=0.2, random_state=4)
print ('Train set:', x_train.shape,  y_train.shape)
print ('Test set:', x_test.shape,  y_test.shape)


nsamples, nx, ny = x_train.shape
x_train = x_train.reshape((nsamples,nx*ny))

nsamples, nx, ny = x_test.shape
x_test = x_test.reshape((nsamples,nx*ny))

print(x_test.shape)
print(x_train.shape)


from sklearn.tree import DecisionTreeClassifier
imageTree = DecisionTreeClassifier(criterion="entropy", max_depth = 2)
imageTree # it shows the default parameters

imageTree.fit(x_train,y_train)
#for train
pred_class_train = imageTree.predict(x_train)
#for test
pred_class_test = imageTree.predict(x_test)


#for train data
print (classification_report(y_train, pred_class_train))

#for test data
print (classification_report(y_test, pred_class_test))


train_accuracy = list()
test_accuracy = list()
max_depth = list()

for depth in range(6,30):
    imageTree = DecisionTreeClassifier(criterion="entropy", max_depth = depth)
    imageTree.fit(x_train,y_train)
    print("for depth :",depth)
    #for train
    pred_class_train = imageTree.predict(x_train)
    print (classification_report(y_train, pred_class_train))
    train_accuracy.append(metrics.accuracy_score(y_train, pred_class_train))
    
    #for test class
    pred_class_test = imageTree.predict(x_test)
    print (classification_report(y_test, pred_class_test))
    
    print(y_test)
    print(pred_class_test)
    train_accuracy.append(metrics.accuracy_score(y_test, pred_class_test))

    max_depth.append(depth)
