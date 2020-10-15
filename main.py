import sift
import kmeans
import os
import cv2 as cv
import numpy as np
import operator
import matplotlib.pyplot as plt
def extract_features(image_paths):
    Features = []  # 所有图像的所有sift descriptor的列表
    for i, image_path in enumerate(image_paths):
        img = cv.imread(image_path)
        gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
        kp, img_des = sift.sift(image_path) 
        Features.append(img_des)
    return Features
def getdata(root_dir):
    #从一个文件夹中获取所有文件的路径，在本项目中是图片
    traind=[]
    for root,dirs,files in os.walk(root_dir):
        for f in files:
            fpath=root
            fpath+="\\"
            fpath+=f
            traind.append(fpath)
    return traind
def featuretoword(Features, center):  
    # 把特征表示的图转化为特征的词频表示
    img_histogram_list = np.zeros((len(Features), len(center)), "float32")
    i=0  
    for img in Features:
        for feature in img:
            labelidx=kmeans.caldis(feature,center)
            img_histogram_list[i][labelidx] += 1 
        i+=1
    img_histogram_list=img_histogram_list/i
    return img_histogram_list


def classify0(inX,dataset,labels,k):
#利用knn求得最相似的几张图像
    datasize=dataset.shape[0]
    diffmat=np.tile(inX,(datasize,1))
    diffmat=diffmat-dataset
    diffmat=diffmat**2
    distancemat=diffmat.sum(axis=1)
    classcount={}
    imat=distancemat.argsort()
    res=[]
    for i in range(k):
        j=imat[i]
        res.append(j)
        classcount[labels[j]]=classcount.get(labels[j],0)+1
    classcount=sorted(classcount.items(),key=operator.itemgetter(1),reverse=True)
    return classcount[0][0],res


def showres(trainDataset,testDataset,similarimg,which_testvec):
    #将结果以图片的形式打印出来
    for i in range(10):
        plt.subplot(3,5,i+6)
        img=cv.imread(trainDataset[similarimg[i]])
        plt.imshow(img,'gray')
        plt.title(str(i))
        plt.xticks([]),plt.yticks([])
    img=cv.imread(testDataset[which_testvec])
    plt.subplot(3,5,1)
    plt.imshow(img,'gray')
    plt.title(str(1))
    plt.xticks([]),plt.yticks([])
    path="res"+str(which_testvec)+".jpg"
    plt.savefig(path)


if __name__ == "__main__":
    if os.path.exists("Features.npy"):
        #如果未提取过训练集的sift特征则提取
        pass
    else :
        root_dir="traindata"
        trainDataset=getdata(root_dir)
        Features=extract_features(trainDataset)
        np.save("Features.npy",Features)
    Features=np.load("Features.npy",allow_pickle=True)
    if os.path.exists("center.npy"):
        #如果未进行kmeans聚类则进行
        pass
    else :
        tempf=[]
        for img in Features:
            for feature in img:
                tempf.append(feature)
        tempf=np.array(tempf)
        center,_=kmeans.kmeans(tempf,1000)
        np.save("center.npy",center)
        model=featuretoword(Features,center)
        np.save("model.npy",model)
    root_dir="testdata"
    center=np.load("center.npy")
    model = np.load("model.npy")
    testDataset=getdata(root_dir)
    root_dir="traindata"
    trainDataset=getdata(root_dir)
    Features=extract_features(testDataset)
    testvec=featuretoword(Features,center)
    np.save("testvec.npy",testvec)
    testvec=np.load("testvec.npy")
    labels=[i for i in range(6)]
    labels=np.array(labels)
    labels=np.tile(labels,5)
    for which_testvec in range(6):
        y_hat,similarimg=classify0(testvec[which_testvec],model,labels,10)
        showres(trainDataset,testDataset,similarimg,which_testvec)
        cntrecall=0
        cntprecision=0
        for imgidx in similarimg:
            if labels[imgidx]==labels[which_testvec]:
                cntrecall+=1
                cntprecision+=1
        cntrecall=cntrecall/5
        cntprecision=cntprecision/10
        print("img:"+str(which_testvec)+" precision:"+str(cntprecision*100)+"% recall:"+str(cntrecall*100)+"%")
