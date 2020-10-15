import math
import cv2 as cv
import matplotlib.pyplot as plt 
import numpy as np 
import os
os.path
os.chdir('E:\pycharmfile\opencv')

def mkG(img,n):
   # 制造金字塔 
    Gspace=[]
    g=img.copy()
    Gspace.append(g)
    for i in range(n):
        g = cv.pyrDown(g)
        Gspace.append(g)
    return Gspace


def mkDOGandAnglespace(img,segma,n):
    #先制造高斯空间，然后逐层相减制造差分高斯空间(DOG)，以及返回第三次迭代产生的空间用于角度计算
    DOGspace=[]
    Gspace=[]
    old_g=[]
    for i in range(n):
        g=img.copy()
        # m 为尺度空间的尺度
        m = int(2**0.5*segma*(i+1)+1)
        if m % 2==0: m+=1
        #利用他的可分性，在x，y方向分别高斯模糊代替二维高斯模糊降低计算量
        g = cv.GaussianBlur(img,(m,1),0)
        g = cv.GaussianBlur(g,(1,m),0)
        if i==0 : 
            old_g=g
            continue
        DOGspace.append( cv.absdiff(g,old_g) )
        old_g = g
        if i==2:
            Gspace=g
    return DOGspace,Gspace


def FindKeypoint(space):
    #寻找关键点，以下赋予数字的地方指的是不考虑边缘点
    #以及尺度空间的顶层和底层
    Keypoint = []
    shape=[len(space),len(space[0]),len(space[0][0])]
    for i in range(1,shape[0]-1):
        for j in range(5,shape[1]-6):
            for k in range(5,shape[2]-6):
                if ifkeypoint(i,j,k,space,shape) and ifedge(i,j,k,space):
                    Keypoint.append([i,j,k])
    return Keypoint


def ifkeypoint(i,j,k,space,shape):
    #判断是否是极值点
    dx=[0,1,-1]
    dy=[0,1,-1]
    dz=[0,1,-1]
    flag=0
    dnow=space[i][j][k]
    for ix in range(3):
        for iy in range(3):
            for iz in range(3):
                if flag==0 :
                    flag=1
                    continue
                if space[i+dx[ix]][j+dy[iy]][k+dz[iz]] >= dnow :
                    return False
    return True


def ifedge(i,j,k,space):
    #判断是否是可去边缘点
    #我们知道边缘点是非常容易产生极值的但有些边缘点不具有特征，故我们将其去除
    dx=space[i][j+1][k]+space[i][j+1][k-1]+space[i][j+1][k+1]
    -(space[i][j-1][k]+space[i][j-1][k-1]+space[i][j-1][k+1])
    dx=abs(dx)
    dy=space[i][j+1][k+1]+space[i][j][k+1]+space[i][j-1][k+1]
    -(space[i][j+1][k-1]+space[i][j][k-1]+space[i][j-1][k-1])
    dy=abs(dy)
    if dy==0 or dx==0: return False
    t=1.0*dx/dy
    if t<10 and t>0.1: 
        return True
    else :
        return False


def showpic(img):
    #展示图片，调试的时候用与实现无关
    cv.imshow('image',img)
    cv.waitKey(0)
    cv.destroyAllWindows()


def angel(Keypoint,space):
    #求取关键点的方向
    i,j,k=Keypoint
    Aspace=space[j-4:j+5,k-4:k+5]
    #求取该方向向量的模
    sobelx=cv.Sobel(Aspace,cv.CV_64F,1,0,ksize=3)
    sobely=cv.Sobel(Aspace,cv.CV_64F,0,1,ksize=3)
    a=sobelx*sobelx
    b=sobely*sobely
    m=a+b
    m=m**0.5
    m=cv.GaussianBlur(m,(9,9),0)
    outm=m[4,4]
    c=a+b
    outa=np.zeros(8)
    #求角度
    for idxj in range(9):
        for idxk in range(9):
            dx=sobelx[idxj,idxk]
            dy=sobely[idxj,idxk]
            nowang=0
            if dx==0 or dy==0:
                continue
            elif dx==0 and dy!=0:
                nowang=0.5*np.pi if dy>0 else 1.5*np.pi 
            elif dx!=0 and dy==0:
                nowang=1*np.pi if dx<0 else 0
            else :
                nowang=np.arctan(dy/dx)
                if dx<0: nowang+=np.pi 
            angidx=int(round(nowang/(0.25*np.pi)))
            outa[angidx % 8]+=1
    outangle=outa.argmax()
    return outm,outangle
    

def getfeature(space,vecangle,Keypoint):
    #求取特征向量
    #旋转坐标轴以实现旋转不变性
    i,j,k=Keypoint
    trans=np.zeros((2,2))
    trans[0,0]=np.cos(vecangle*np.pi/4)
    trans[1,1]=np.cos(vecangle*np.pi/4)
    trans[0,1]=-np.sin(vecangle*np.pi/4)
    trans[1,0]=np.sin(vecangle*np.pi/4)
    #获得附近点
    Aspace = space[j-4:j+5,k-4:k+5]
    locate=np.zeros((2,1))
    features=np.zeros((4,4,8))
    for x in range(-4,5):
        for y in range(-4,5):
            locate[0][0]=x
            locate[1][0]=y
            locate=np.matmul(trans,locate)
            newx=locate[0,0]
            newy=locate[1,0]
            if newx<-2 or newx>=2 or newy<-2 or newy>=2: continue
            newx=math.floor(newx)
            newy=math.floor(newy)
            #计算八方向梯度
            #以 0，1，2排序
            #   3， ，4
            #   5，6，7
            #以在原图的方向减去旋转的角度得到
            dx=[1,0,-1]
            dy=[1,0,-1]
            idx=0
            for a in dx:
                for b in dy:
                    if a==0 and b==0 : continue
                    
                    dd= int(Aspace[newx+4+a,newy+4+b])-int(Aspace[newx+4,newy+4])
                    features[newx+2][newy+2][idx]=dd
                    idx+=1
    return features.reshape(-1)

            
def showpoint(Keypoint,img):
    for point in Keypoint:
        cv.circle(img,(point[1],point[2]), 3, (0,0,255), 1)


def sift(img_path):
    #sift特征提取
    img=cv.imread(img_path)
    img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    space = mkG(img,7)
    G_DOGspace = []
    g=img.copy()
    idx=1
    res=[]
    kp=[]
    for pic in space:
        DOGspace,Aspace=mkDOGandAnglespace(pic,5,7)
        Keypoints = FindKeypoint(DOGspace)
        kp.append(Keypoints)
        for point in Keypoints:
            vecm,vecagle = angel(point,Aspace)
            feature= getfeature(Aspace,vecagle,point)
            res.append(feature)
    return kp,res