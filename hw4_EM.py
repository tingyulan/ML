import numpy as np
import os
import math
import matplotlib.pyplot as plt
import numba as nb
from numba import autojit


def readmnist(mnist_dir, mode='training'):
    if mode == 'training':
        image_dir = os.path.join(mnist_dir, 'train-images-idx3-ubyte')
        label_dir = os.path.join(mnist_dir, 'train-labels-idx1-ubyte')
    elif mode == 'testing':
        image_dir = os.path.join(mnist_dir, 't10k-images-idx3-ubyte')
        label_dir = os.path.join(mnist_dir, 't10k-labels-idx1-ubyte')

    with open(image_dir, 'rb') as fimage:
        #training: magic=2049, num=60000, row=28, col=28
        magic, num, row, col = np.fromfile(fimage, dtype=np.dtype('>i'), count=4)
        images = np.fromfile(fimage, dtype=np.dtype('>B'), count=-1)

    with open(label_dir, 'rb') as flabel: #read binary mode
        magic, num = np.fromfile(flabel, dtype=np.dtype('>i'), count=2)  #big endian, int
        labels = np.fromfile(flabel, dtype=np.dtype('>B'), count=-1) #unsigned byte, realall

    pixels = row*col
    images = images.reshape(num, pixels)

    return num, images, labels, pixels

@autojit
def ExpectationStep(p, pi, eta, X):
    for i in range(60000):
        for k in range(10):
            eta[i,k] = pi[k,0]
            for d in range (784):
                if X[i,d]==1:
                    eta[i,k] *= p[k,d]
                else:
                    eta[i,k] *= (1-p[k,d])
        normalize = np.sum(eta[i,:])
        if normalize!=0:
            eta[i,:] /= normalize
    return eta

@autojit
def MaximizationStep(p, pi, eta, X):
    a1 = 1e-8
    a2 = 1e-8
    Nk = np.zeros(10)
    for k in range(10):
        for i in range (60000):
            Nk[k] += eta[i,k]
    
    for k in range(10):
        #print('k:', k)
        for d in range(784):
            p[k,d] = 0
            for i in range(60000):
                p[k,d] += eta[i,k]*X[i,d]
            p[k,d] = (p[k,d]+a1)/(Nk[k]+a2*D)
        pi[k,0] = (Nk[k]+a1)/(np.sum(Nk)+a2*K)
    return p, pi

def show_imageination(p, count, diff):
    im = np.zeros((10, 784))
    im = (p>=0.5)*1
    for c in range(K):
        print("clustering class {}:".format(c))
        for row in range (28):
            for col in range(28):
                print(im[c][row*28+col], end=' ')
            print(" ")
        print("")

    print("No. of Iteration: {}, Difference: {}".format(count, diff))
    print("-----------------------------------------\n")

def final_imagination(p, predict_gt_relation):
    im = (p>=0.5)*1
    for c in range(10):
        for k in range(10):
            if predict_gt_relation[k]==c:
                choose = k
        print("labeled class {}:".format(c))
        for row in range (28):
            for col in range(28):
                print(im[choose][row*28+col], end=' ')
            print(" ")
        print("")
    print('-----------------------------------------\n')

@autojit
def make_prediction(X, p):
    predict_gt = np.zeros((10,10))
    pdistribution = np.zeros(10)
    for i in range(60000):
        for k in range(10):
            pdistribution[k] = pi[k,0]
            for d in range(784):
                if X[i][d] == 1:
                    pdistribution[k] *= p[k][d]
                else:
                    pdistribution[k] *= (1 - p[k][d])
        predict = np.argmax(pdistribution)
        predict_gt[predict, train_labels[i]]+=1
    return predict_gt

@autojit
def remake_prediction(X, p, predict_gt_relation):
    predict_gt = np.zeros((10,10))
    pdistribution = np.zeros(10)
    for i in range(60000):
        for k in range(10):
            pdistribution[k] = pi[k,0]
            for d in range(784):
                if X[i][d] == 1:
                    pdistribution[k] *= p[k][d]
                else:
                    pdistribution[k] *= (1 - p[k][d])
        pred = np.argmax(pdistribution)
        predict_gt[predict_gt_relation[pred], train_labels[i]]+=1
    return predict_gt

def shift_cluster(predict_gt):
    predict_gt_relation = np.full((10), -1, dtype=np.int)
    for k1 in range(10):
        ind = np.unravel_index(np.argmax(predict_gt, axis=None), (10,10)) #gt, pred
        predict_gt_relation[ind[0]] = ind[1]
        for k2 in range(10):
            predict_gt[ind[0]][k2] = -1
            predict_gt[k2][ind[1]] = -1
    # print(predict_gt_relation)
    return predict_gt_relation

@autojit
def calculate_confusion(k, predict_gt):
    tp=fn=fp=tn=0
    for pred in range(K):
        for tar in range(K):
            if pred==k and tar==k:
                tp += predict_gt[pred, tar]
            elif pred==k:
                fp += predict_gt[pred, tar]
            elif tar==k:
                fn += predict_gt[pred, tar]
            else:
                tn += predict_gt[pred, tar]
    return int(tp), int(fn), int(fp), int(tn)

def confusion(predict_gt, count):
    hit = 60000
    for k in range(K):
        tp, fn, fp, tn = calculate_confusion(k, predict_gt)
        hit -= tp
        print('Confusion Matrix {}:'.format(k))
        print('{:^20}{:^25}{:^25}'.format(' ', 'Predict number %d'%k, 'Predict not number %d'%k))
        print('{:^20}{:^25}{:^25}'.format('Is number %d'%k, tp, fn))
        print('{:^20}{:^25}{:^25}\n'.format('Isn\'t number %d'%k, fp, tn))
        print('Sensitivity (Successfully predict number {}):     {}'.format(k, tp/(tp+fn)))
        print('Specificity (Successfully predict not number {}): {}'.format(k, tn/(fp+tn))) #fp/(fp+tn)
    print('Total iteration to converge:', count)
    print('Total error rate:', hit/60000)


if __name__=='__main__':
    #--------prepare data----------
    mnist_dir = './data/'
    train_num, train_images, train_labels, num_pixels = readmnist(mnist_dir, 'training') #60000, 60000*784, 60000, 784
    N = train_num
    D = num_pixels #784
    K = 10
    X = np.zeros(train_images.shape, dtype=int) #60000*784
    for r in range(60000):
        for c in range(784):
            if(train_images[r,c]>=128):
                X[r,c]=1
    np.save('./npy/X.npy', X)
    # X = np.load("./npy/X.npy")
    #----------initial---------      
    eta = np.zeros((train_num,K)) #numpy.float64
    p = np.random.uniform(0.0, 1.0, (10,784))
    for k in range(K):
        tmp = np.sum(p[k,:])
        p[k,:] /= tmp
    np.save('./npy/p_init.npy', p)
    # p = np.load("./npy/p_init_502716.npy")
    pi = np.full((10,1), 0.1)
    # predict_gt = np.zeros((10,10)) 
    #-----------EM algorithm--------------------
    count = 0
    while(True):
        p_old = p
        count += 1
        eta = ExpectationStep(p, pi, eta, X)
        p, pi = MaximizationStep(p, pi, eta, X)
        show_imageination(p, count, np.linalg.norm(p-p_old))
        if(count==20 and np.linalg.norm(p-p_old)<1e-10):
            break
    
    predict_gt = make_prediction(X, p)
    predict_gt_relation = shift_cluster(predict_gt)
    predict_gt = remake_prediction(X, p, predict_gt_relation)
    final_imagination(p, predict_gt_relation)
    confusion(predict_gt, count)
        
