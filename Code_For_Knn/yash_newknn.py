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

def limit(data):
#code for taking class 2 only
    i=0
    j=0
    k=0
    democlasses=[]
    demoimage=[]
    
    while(i!=245):
      if(data[j][1]==2):
        democlasses.append(data[j][1])
        demoimage.append(data[j][0])
        i+=1
        j+=1
      else:
        j+=1
    
    #code for taking class 1 only
    i=0
    j=0
    while(i!=245):
      if(data[j][1]==1):
        democlasses.append(data[j][1])
        demoimage.append(data[j][0])
        i+=1
        j+=1
      else:
        j+=1
    
    #code for taking class 3 only
    i=0
    j=0
    while(i!=245):
      if(data[j][1]==3):
        democlasses.append(data[j][1])
        demoimage.append(data[j][0])
        i+=1
        j+=1
      else:
        j+=1
    
    
    #code for taking class 4 only
    i=0
    j=0
    while(i!=245):
      if(data[j][1]==4):
        democlasses.append(data[j][1])
        demoimage.append(data[j][0])
        i+=1
        j+=1
      else:
        j+=1
        
        
    return democlasses,demoimage


democlasses,demoimage=limit(data)


image=[]
classes=[]


#taking image from csv file
for i in range(980):   
  name=demoimage[i]
  join_path=os.path.join(path2,name)  
  img=cv2.imread(join_path)#read 5000 setof image
  img2=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
  image.append(img2)#append all the image in imge
  classes.append(democlasses[i])#append all the class in classes



#converitng into numpy
image=np.array(image)
classes=np.array(classes)

nsamples, nx, ny = image.shape
image = image.reshape((nsamples,nx*ny))

from sklearn import preprocessing
image = preprocessing.StandardScaler().fit(image).transform(image.astype(float))


#split train and test
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(image, classes, test_size=0.2, random_state=5)
print ('Train set:', x_train.shape,  y_train.shape)
print ('Test set:', x_test.shape,  y_test.shape)






print(x_test.shape)
print(x_train.shape)



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