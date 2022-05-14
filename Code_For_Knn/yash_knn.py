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



from sklearn import preprocessing
image = preprocessing.StandardScaler().fit(x_train).transform(x_train.astype(float))


#Classifier implementing the k-nearest neighbors vote.
from sklearn.neighbors import KNeighborsClassifier

#take k=4
k = 4
#Train Model and Predict  
neigh = KNeighborsClassifier(n_neighbors = k).fit(x_train,y_train)


#for train
pred_class_train = neigh.predict(x_train)
print(y_train)
print(pred_class_train)

print (classification_report(y_train, pred_class_train))



#for test
pred_class_test = neigh.predict(x_test)
print(y_test)
print(pred_class_test)
print (classification_report(y_test, pred_class_test))


Ks = 10
mean_acc = np.zeros((Ks-1))
std_acc = np.zeros((Ks-1))

for n in range(1,Ks):
    
    #Train Model and Predict  
    neigh = KNeighborsClassifier(n_neighbors = n).fit(x_train,y_train)
    yhat=neigh.predict(x_test)
    mean_acc[n-1] = metrics.accuracy_score(y_test, yhat)
    std_acc[n-1]=np.std(yhat==y_test)/np.sqrt(yhat.shape[0])

print(mean_acc)



print( "The best accuracy was with", mean_acc.max(), "with k=", mean_acc.argmax()+1)