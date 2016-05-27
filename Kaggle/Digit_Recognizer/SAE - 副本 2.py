import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib as mpl
import random as rd
import gzip
import cPickle
from sklearn import svm
def readMat(path):
    f=gzip.open(path,'rb')
    train_set,valid_set,test_set=cPickle.load(f)
    f.close()
    return train_set,valid_set,test_set

def imageShow(data):
    plt.close()
    cmap = mpl.cm.gray_r
    l=9
    for i in range(l):
        plt.sca(plt.subplot(521+i))
        plt.imshow(np.reshape(data[i,:],(28,28)),cmap=cmap)
    plt.show()
    plt.close()
def normalizeData(patches):
    ma=patches.max()
    mi=patches.min()
    patches = (patches-mi)/(ma-mi)
 #   patches =patches * 2 -1
    patches=patches*0.8+0.1
    return patches

def initializeParameters(hiddenSize,visibleSize):
    theta=[]
    W1=np.random.random(size=hiddenSize*(visibleSize));
    W1=np.reshape(W1,(hiddenSize,visibleSize))

    W2=np.random.random(size=hiddenSize*(visibleSize));
    W2=np.reshape(W2,(visibleSize,hiddenSize))

    b1=np.random.random(size=hiddenSize)
    b1=np.reshape(b1,(hiddenSize,1))
    b2=np.random.random(size=visibleSize)
    b2=np.reshape(b2,(visibleSize,1))

    r=np.sqrt(6)/np.sqrt(hiddenSize+visibleSize+1)
    W1=W1*2*r-r
    W2=W2*2*r-r

    return np.mat(W1),np.mat(W2),np.mat(b1),np.mat(b2)

def FWandBPandUP(w1,w2,b1,b2,data,alpha,labda,m,beta,sparsity_param):

    #print w1.shape,w2.shape,b1.shape,b2.shape,data.shape

    z2=w1*data+b1
    a2=sigmoid(z2)

    z3=w2*a2+b2
    a3=sigmoid(z3)

    rho_hat=np.sum(a2,axis=1)/m
    rho_hat=np.mat(rho_hat)
    rho=np.tile(sparsity_param,(w1.shape[0],1))
    rho=np.mat(rho)

    costj=0.5/m*(sum(sum(np.multiply(a3-data,a3-data)).T)[0,0])+\
           labda/2*(sum(sum(np.multiply(w1,w1)).T)[0,0]+\
                    sum(sum(np.multiply(w2,w2)).T)[0,0])+\
                    beta*(sum(KL_divergence(rho,rho_hat))[0,0])

    sparsity_delta=-rho/rho_hat+(1-rho)/(1-rho_hat)
    sparsity_delta=np.mat(sparsity_delta)

    kethe3=-np.multiply(data-a3,sigmoidInv(z3))
    kethe2=np.multiply(np.mat(w2).T*kethe3+beta*sparsity_delta,sigmoidInv(z2))
    #kethe2=np.multiply(np.mat(w2).T*kethe3,np.multiply(a2,1-a2))

    deltaw2=kethe3*a2.T
    deltaw1=kethe2*data.T

    deltab1=sum(kethe2.T).T
    deltab2=sum(kethe3.T).T

    w1=w1-alpha*((1.0/m*deltaw1)+labda*w1)
    w2=w2-alpha*((1.0/m*deltaw2)+labda*w2)

    b1=b1-alpha*(1.0/m*deltab1)
    b2=b2-alpha*(1.0/m*deltab2)
    return w1,w2,b1,b2,costj


def sigmoid(data):
    return 1.0/(1+np.exp(-data))

def sigmoidInv(data):
    return np.multiply(sigmoid(data),(1-sigmoid(data)))

def KL_divergence(x, y):
    return np.multiply(x , np.log(x / y)) + np.multiply((1 - x) , np.log((1 - x) / (1 - y)))

def shared_dataset(data_xy):
    data_x,data_y=data_xy
    return data_x,data_y

def pooling_max(data,m,n,po):
    data.reshape((len(data),m,n))
    a=[[[0 for i in range(n/po)] for j in range(m/po)] for k in range(len(data))]

    for i in range(len(data)):
        for j in range(m):
            for k in range(n):
                t=data[i][j:j+po][k:k+po].argmax()
                a[i][j%po][k%po]=t
                k+=po
            j+=po
    return a

def layer1(train_set_x,test_set_x,vis,hid,ste):

    data=np.mat(train_set_x)[0:20000][:]
    #print data.shape
    batches=20
    visibleSize=vis
    hiddenSize=hid
    labda = 0.0001
    alpha=1.0
    m=data.shape[0]/batches
    beta=0.1
    sparsity_param=0.5

    w1,w2,b1,b2=initializeParameters(hiddenSize,visibleSize)
    b1[:]=1.0
    b2[:]=1.0

    #print w1.shape,w2.shape,b1.shape,b2.shape

    steps=ste
    for i in range(steps):
        w1,w2,b1,b2,costj=FWandBPandUP(w1,w2,b1,b2,data[i%10*m:i%10*m+m].T,alpha,labda,m,beta,sparsity_param)

        if i%10==0:
            print i,' : ',costj

    #a3=sigmoid(w2*sigmoid(w1*data.T+b1)+b2).T
    a2=sigmoid(w1*data.T+b1)
    a2_test=sigmoid(w1*test_set_x.T+b1)
    return a2,a2_test,w1,w2,b1,b2

def layer2(data,test_data,vis,hid,ste):

    batches=20
    visibleSize=vis
    hiddenSize=hid
    labda = 0.0001
    alpha=1.0
    m=data.shape[0]/batches
    beta=0.1
    sparsity_param=0.5

    w1,w2,b1,b2=initializeParameters(hiddenSize,visibleSize)
    b1[:]=1.0
    b2[:]=1.0

    steps=ste
    for i in range(steps):
        w1,w2,b1,b2,costj=FWandBPandUP(w1,w2,b1,b2,data[i%10*m:i%10*m+m].T,alpha,labda,m,beta,sparsity_param)

        if i%10==0:
            print i,' : ',costj

    a2=sigmoid(w1*data.T+b1)
    a2_test=sigmoid(w1*test_data+b1)
    a3=sigmoid(w2*a2+b2).T
    return a3,a2,a2_test,w1,w2,b1,b2

def traintest_num(train_x,train_y,test_x,test_y):
    clf=svm.SVC()
    clf.fit(train_x,train_y)
    cr=0
    wr=0
    for i in range(len(test_x)):
        if clf.predict(test_x[i])[0] ==test_y[i]:
                   cr+=1
        else:
                   wr+=1
    return cr,wr
   

if __name__=='__main__':
    path="/Users/Ximo/Documents/workspace/CNN/mnist.pkl.gz";
    train_set,valid_set,test_set=readMat(path);
    test_set_x,test_set_y=shared_dataset(test_set)
    valid_set_x,valid_set_y=shared_dataset(valid_set)
    train_set_x,train_set_y=shared_dataset(train_set)
    
    #data,test_data,w11,w12,b11,b12=layer1(train_set_x,test_set_x,28*28,200,400)
#
#    a3,a2,a2_test,w21,w22,b21,b22=layer2(data.T,test_data,200,10,400)
#    
#    cr,wr=traintest_num(a2.T,train_set_y[0:20000],a2_test.T,test_set_y)
