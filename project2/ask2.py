import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random
import math
from sklearn import metrics

fashion_mnist = tf.keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()


trueCat = 0


def initialize_centers():
    global train_images
    #arxikopoioume ta 10 kentra mas,osa dhladh kai oi kathgories pou 8eloume na xwristoun ta dedomena mas
    centers = []
    for i in range(0,10):
        centers.append([0]*784)

    #vazoume tuxaious ari8mous gia ka8e ena apo ta 10 kentra mas.
    for i in range(0,10):
        for j in range(0,784):
            centers[i][j] = random.randint(0, 255)/255.0
    

    #allazoume to sxhma tou ka8e pinaka apo 28x28 se 784x1
    train_images = train_images.reshape(len(train_images),-1)
    #print(train_images.shape)
    return centers


def changeWeight(a,d,centers,arrayPic,bestCenter):
    start = bestCenter - d
    end = bestCenter + d
    
    if start < 0:
        start = 0
    if end > 9 :
        end = 9
    for j in range(start,end+1):
        center = centers[j]
        for i in range(0,784):
            center[i] = center[i]+ a*(arrayPic[i]-center[i])
        centers[j] = center
    return centers

def euclidian_distance(a,d,centers,arrayPic):
    bestDic = 1000000
    for i in range(0,10):
        center = centers[i]
        dist = np.sqrt(np.sum(np.square(center-arrayPic)))
        if dist < bestDic:
            bestDic = dist
            bestCenter = i
    
    centers= changeWeight(a,d,centers,arrayPic,bestCenter)
    return centers , bestCenter , bestDic


def countMostNumber(array):
    global trueCat
    BestNum = 0
    BestCat = -1
    for i in range(0,10):
        count = 0
        for j in range(0,len(array)):
            if(array[j] == i):
                count = count +1
        if count>BestNum:
            BestNum = count
            BestCat = i
    trueCat = trueCat + BestNum
    #print("to pososto twn swsta taxinomhmenwn gia th kathgoria", BestCat," einai: ",(BestNum/len(array))*100)
    return BestCat

def findTpFpFn(predCat,originalCat):
    catArr = []
    for i in range(0,len(predCat)):
        catArr.append(originalCat[i])
    fmeasure = metrics.f1_score(catArr,predCat,average="weighted")
    print("f1 score: ",fmeasure)
    
        
def main():
    global train_images
    astart = 0.5
    dstart = 5
    trainImagesLen = 1000
    category = [0]*trainImagesLen
    train_images = train_images/255.0
    centers = initialize_centers()
    totalRepeat = 20
    print("Gia ",trainImagesLen," eikones kai ",totalRepeat," epanalhpshs")
    for x in range(0,totalRepeat):
        #print("Repeat number: ",x)
        a = astart*(1-x/totalRepeat)
        d = int(dstart*(1-x/totalRepeat))
        #print("Eimaste sto vhma: ",x)
        for i in range(0,trainImagesLen):
            arrayRet = euclidian_distance(a,d,centers, train_images[i])
            centers = arrayRet[0]
            category[i] = arrayRet[1]
            #print("best Center: ",arrayRet[1],"Dic: ",arrayRet[2],"right category: ",train_labels[i])
    for j in range(0,10):
        count = 0
        helpArray = []
        for k in range(0,trainImagesLen):
            if(category[k]==j):
                helpArray.append(train_labels[k])
                count = count + 1
        num = countMostNumber(helpArray)
        for l in range(0,len(category)):
            if category[l] == j:
                category[l] = num
    
    findTpFpFn(category, train_labels)

    print("To sunoliko Purity einai: ",(trueCat/trainImagesLen)*100,"%")


main()


