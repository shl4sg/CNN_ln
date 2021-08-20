
from scipy import ndimage
import numpy as np
import os
import pandas as pd
rootpath="./img2/"
trainpath=rootpath+"train/"
testpath=rootpath+"test/"
trainlist=os.listdir(trainpath)
testlist=os.listdir(testpath)


def dataNormalize(npzarray):
    maxHU = 1000.  # maxHU = np.amax(npzarray)
    minHU = -1000.  # minHU = np.amin(npzarray)
    npzarray = (npzarray - minHU) / (maxHU - minHU)
    npzarray[npzarray > 1] = 1.
    npzarray[npzarray < 0] = 0.
    return npzarray

def img_resize(img):
    height,width=img.shape
    zoomh=64/height if height>64 else 1
    zoomw=64/width if width>64 else 1
    img= ndimage.zoom(img, zoom=(zoomh,zoomw))
    return  img

def img_pad(img,length):
    xx,yy=img.shape
    xl=(length-xx)//2
    xr=length-xx-xl
    yu=(length-yy)//2
    yb=length-yy-yu
    img = np.pad(img, ((xl, xr), (yu, yb)), 'constant', constant_values=(-1000, -1000))
    return  img



def datagen():
    df = pd.read_excel('ln.xlsx')
    names = df['name']
    labels = df['label']
    Y_train=[]
    Y_tests=[]
    Y_testss = []
    X_train = []
    X_tests=[]
    X_testss=[]

    for fold in testlist:
        X_test = []
        Y_test = []
        for i in range(len(names)):
            if (names[i][:-4] == fold):
                label = labels[i]
        foldpath = testpath + fold
        imglist = os.listdir(foldpath)
        for p in imglist:

            img = np.load(foldpath + "/" + p)
            img = img_resize(img)
            img=img_pad(img,64)
            img = dataNormalize(img)
            img = np.reshape(img, (64, 64, 1))
            X_test.append(img)
            X_testss.append(img)
            if (label == 'A'):
                Y_test.append(0)
                Y_testss.append(0)
            else:
                Y_test.append(1)
                Y_testss.append(1)
        Y_test = np.array(Y_test, dtype='int32')
        X_test = np.array(X_test, dtype='int32')
        Y_tests.append(Y_test)
        X_tests.append(X_test)

    for fold in trainlist:
        foldpath = trainpath + fold
        imglist = os.listdir(foldpath)
        for i in range(len(names)):
            if (names[i][:-4] == fold):
                label = labels[i]
        for p in imglist:
            img = np.load(foldpath + "/" + p)
            img = img_resize(img)
            img=img_pad(img,64)
            img = dataNormalize(img)
            img = np.reshape(img, (64, 64, 1))
            X_train.append(img)
            if (label == 'A'):
                Y_train.append(0)
            else:
                Y_train.append(1)

    for fold in trainlist:
        foldpath = trainpath + fold
        imglist = os.listdir(foldpath)
        for i in range(len(names)):
            if (names[i][:-4] == fold):
                label = labels[i]
        for p in imglist:
            img = np.load(foldpath + "/" + p)
            img = img_resize(img)
            img=img_pad(img,64)
            img = dataNormalize(img)
            img = np.reshape(img, (64, 64, 1))
            if (label == 'A'):
                X_train.append(img)
                Y_train.append(0)



    Y_train = np.array(Y_train, dtype='int32')
    Y_testss=np.array(Y_testss,dtype='int32')
    X_testss=np.array(X_testss, dtype='int32')
    X_train = np.array(X_train, dtype='int32')
    return X_train,Y_train,X_tests,Y_tests,X_testss,Y_testss





