import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import sklearn.neighbors
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn import datasets
from sklearn import metrics
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.linear_model import LogisticRegression
from math import exp
import scipy.special
from scipy.special import softmax
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import GaussianNB

def Nearest_Neighbor(k,methodNum):
    if methodNum == 1:
        print("Erwthma A cosine ",k)
        method = 'cosine'
    elif methodNum == 2:
        print("Erwthma B eucliadian me ",k)
        method = 'euclidean'
    else:
        print("put 1 for cosine\nput 2 for euclidian")
        exit()
    fashion_mnist = tf.keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    nsamples, nx, ny = train_images.shape
    nsamplestest, nxtest, nytest = test_images.shape
    train_images = train_images.reshape((nsamples,nx*ny))
    test_images = test_images.reshape((nsamplestest,nxtest*nytest))
    model = KNeighborsClassifier(n_neighbors=k , metric = method)
    model.fit(train_images , train_labels)
    pred = model.predict(test_images)
    print("Accuracy:" , metrics.accuracy_score(test_labels , pred))
    print('F1 Score: ', metrics.f1_score(test_labels, pred , average = 'weighted'))

def Neural_Networks(erwthma):
    fashion_mnist = tf.keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    nsamples, nx, ny = train_images.shape
    nsamplestest, nxtest, nytest = test_images.shape
    train_images = train_images.reshape((nsamples,nx*ny))
    test_images = test_images.reshape((nsamplestest,nxtest*nytest))
    train_images = train_images / 255.0
    test_images = test_images / 255.0
    if erwthma == 1:
        print("Erwthma A")
        model2 = MLPClassifier(hidden_layer_sizes = 500 ,activation = 'logistic' , solver = "sgd" )
    elif erwthma == 2:
        print("Erwthma B")
        model2 = MLPClassifier(hidden_layer_sizes = (500,200) ,activation = 'logistic' , solver = "sgd" )
    else:
        print("1 gia erwthma a\n2 gia erwthma b")
        exit()
    model2.fit(train_images , train_labels)
    pred = model2.predict(test_images)
    pred2 = model2.predict_proba(test_images)
    soft = softmax(pred2)
    print("pithanothta: " , soft.sum())
    print('Accuracy: ', metrics.accuracy_score(test_labels, pred))
    print('F1 Score: ', metrics.f1_score(test_labels, pred , average = 'weighted'))

def SVM(erwthma):
    if erwthma == 1:
        print("Erwthma A linear")
        kernelFunc = "linear"
    elif erwthma == 2:
        print("Erwthma B Gaussian")
        kernelFunc = "rbf"
    else:
        print("1 gia linear\n2 gia rbf")
        exit()
    fashion_mnist = tf.keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    nsamples, nx, ny = train_images.shape
    nsamplestest, nxtest, nytest = test_images.shape
    train_images = train_images.reshape((nsamples,nx*ny))
    test_images = test_images.reshape((nsamplestest,nxtest*nytest))
    train_images = train_images / 255.0
    test_images = test_images / 255.0
    svc = SVC(kernel=kernelFunc)
    ovall = OneVsRestClassifier(svc)
    ovall.fit(train_images , train_labels)
    pred = ovall.predict(test_images)
    print('Accuracy: ', metrics.accuracy_score(test_labels, pred))
    print('F1 Score: ', metrics.f1_score(test_labels, pred , average = 'weighted'))

def Naive_Bayes_classifier():
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
            'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    fashion_mnist = tf.keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    nsamples, nx, ny = train_images.shape
    nsamplestest, nxtest, nytest = test_images.shape
    train_images = train_images.reshape((nsamples,nx*ny))
    test_images = test_images.reshape((nsamplestest,nxtest*nytest))
    train_images = train_images / 255.0
    test_images = test_images / 255.0
    model2 = GaussianNB()
    model2.fit(train_images , train_labels)
    pred = model2.predict(test_images)
    print('Accuracy' , metrics.accuracy_score(test_labels, pred))
    print('F1 Score' , metrics.f1_score(test_labels, pred , average = 'weighted'))

if __name__=="__main__":
    #Nearest_Neighbor(10,2)#neighbor number , method number(1 for cosine,2 for euclidean)
    #Neural_Networks(2) #1 gia erwthma a kai 2 gia erwthma b
    SVM(2)# 1 gia linear , 2 gia Gaussian
    #Naive_Bayes_classifier()