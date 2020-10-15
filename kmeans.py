import numpy as np
def caldis(inX,center):
    #计算inX与各个类核心的距离
    d=np.tile(inX,(len(center),1))
    distance=d-center
    distance=distance**2
    distance=distance.sum(1)
    distance=distance**0.5
    dividx=distance.argmin(0)
    return dividx
def kmeans(dataset,k):
    #实现kmeans聚类
    lenofdata=dataset.shape[0]
    np.random.shuffle(dataset)
    i=0
    lenofx=dataset.shape[1]
    center=np.copy(dataset[:k,:])
    newcenter=np.array([])
    divide=[[i] for i in range(k)]
    flag=0
    while 1 :
    
        newcenter=newcenter.tolist()
        if flag==0: 
            starti=k
            flag=1
        else : 
            starti=0
            divide=[[]for i in range(k)]
            newcenter=[]
        for i in range(starti,lenofdata):
            #计算距离
            dividx=caldis(dataset[i],center)
            divide[dividx].append(i)
        #判断分类是否合适，合适则停止否则继续
        for line in divide:
            sumd=0
            cnt=0
            for d in line:
                sumd=dataset[d]+sumd
                cnt+=1
            newcenter.append(sumd/cnt)
        newcenter=np.array(newcenter)
        bias=newcenter-center
        bias=bias**2
        bias=bias.sum(0)
        bias=bias.sum(0)
        bias=bias**0.5
        center=center**2
        center=center.sum(0)
        center=center.sum(0)
        center=center**0.5
        #当误差小于0.05时返回，视实际需求可调整
        if bias/center<0.03: break
        else : center=newcenter    
    return newcenter,divide



