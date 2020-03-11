import numpy as np
import math
import matplotlib.pyplot as plt

def univariate_gaussian(m=3.0, s=5.0):
    #Central Limit Theorem
    X = np.sum(np.random.uniform(0.0, 1.0, 12))-6
    return X * s**0.5 + m

def visualize():
    g_class1=[]
    g_class2=[]
    gtp=gfp=gfn=gtn=0
    for i in range(len(X)):
        if X[i].dot(w_gradient)>=0:
            g_class1.append(X[i, 0:2])
            if Y[i,0]==1:
                gtp+=1
            else:
                gfp+=1
        else:
            g_class2.append(X[i, 0:2])
            if Y[i,0]==0:
                gtn+=1
            else:
                gfn+=1
    g_class1 = np.array(g_class1)
    g_class2 = np.array(g_class2)

    n_class1=[]
    n_class2=[]
    ntp=nfp=nfn=ntn=0
    for i in range(len(X)):
        if X[i].dot(w_newton)>=0:
            n_class1.append(X[i, 0:2])
            if Y[i,0]==1:
                ntp+=1
            else:
                nfp+=1
        else:
            n_class2.append(X[i, 0:2])
            if Y[i,0]==0:
                ntn+=1
            else:
                nfn+=1
    n_class1 = np.array(n_class1)
    n_class2 = np.array(n_class2)

    print('Gradient descent:\n')
    print('w:\n',w_gradient[0,0],'\n',w_gradient[1,0],'\n', w_gradient[2,0])
    print('Confusion Matrix:')
    print('\t\t\t Is cluster 1\t Is cluster 2')
    print('Predict cluster 1\t   ', gtp, '\t\t   ', gfn)
    print('Predict cluster 2\t   ', gfp, '\t\t   ', gtn)
    print('\nSensitivity (Successfully predict cluster 1): ', gtp/(gtp+gfn))
    print('Specificity (Successfully predict cluster 2): ', gtn/(gtp+gfn))

    print('\n-------------------')
    print('Newton\'s method\n')
    print('w:\n',w_newton[0,0],'\n',w_newton[1,0],'\n', w_newton[2,0])
    print('Confusion Matrix:')
    print('\t\t\t Is cluster 1\t Is cluster 2')
    print('Predict cluster 1\t   ', ntp, '\t\t   ', nfn)
    print('Predict cluster 2\t   ', nfp, '\t\t   ', ntn)
    print('\nSensitivity (Successfully predict cluster 1): ', ntp/(ntp+nfn))
    print('Specificity (Successfully predict cluster 2): ', ntn/(ntp+nfn))

    #ground truth
    plt.figure()
    plt.subplot(131)
    plt.scatter(c1[:,0], c1[:,1], c='r')
    plt.scatter(c2[:,0], c2[:,1], c='b')

    #gradient descent
    plt.subplot(132)
    if len(g_class1)!=0:
        plt.scatter(g_class1[:,0], g_class1[:,1], c='b')
    if len(g_class2)!=0:
        plt.scatter(g_class2[:,0], g_class2[:,1], c='r')

    plt.subplot(133)
    if len(n_class1)!=0:
        plt.scatter(n_class1[:,0], n_class1[:,1], c='b')
    if len(n_class2)!=0:
        plt.scatter(n_class2[:,0], n_class2[:,1], c='r')

    plt.tight_layout()
    plt.show()

def first_derivative(X, Y, w):
    return X.T.dot( Y -(1/(1+np.exp(-X.dot(w)))) )


def gradient_decent():
    w = np.random.rand(3,1)
    # print("initial:", w.T)
    count = 0
    while(True):
        count += 1
        w_old = w
        deltaJ = first_derivative(X, Y, w)
        w = w + deltaJ
        if(np.linalg.norm(w-w_old)<1e-2 or (count>1e4 and np.linalg.norm(w-w_old)<80) or count>1e5):
            break
    return w

def newton():
    w = np.random.rand(3,1)
    D = np.zeros((N*2, N*2))
    count = 0
    while(True):
        count += 1
        w_old = w
        for i in range(N*2):
            e = np.exp(-X[i].dot(w))
            if math.isinf(e):
                e = np.exp(100)

            D[i][i] = e / ((1+e)**2)
            
        
        H = X.T.dot(D.dot(X))
        deltaf = first_derivative(X,Y,w)
        if np.linalg.det(H)==0:
            w = w + deltaf
        else:
            w = w + np.linalg.inv(H).dot(deltaf)
        
        if(np.linalg.norm(w-w_old)<1e-2 or (count>1e4 and np.linalg.norm(w-w_old)<5) or count>1e5):
            break
        
    return w




if __name__=='__main__':
    #-------------INPUT-----------------
    N = 50
    mx1=my1=1
    mx2=my2=3
    vx1=vy1=2
    vx2=vy2=4
    #-------------END INPUT-------------

    c1 = np.zeros((N,2))
    c2 = np.zeros((N,2))

    for i in range(N):
        c1[i,0] = univariate_gaussian(mx1, vx1)
        c1[i,1] = univariate_gaussian(my1 ,vy1)
        c2[i,0] = univariate_gaussian(mx2, vx2)
        c2[i,1] = univariate_gaussian(my2, vy2)
    
    X = np.ones((N*2, 3))
    X[0:N,0:2] = c1
    X[N:N*2,0:2] = c2
    Y = np.zeros((N*2, 1), dtype=int)
    Y[N:N*2,0] = 1

    w_gradient = gradient_decent()
    w_newton = newton()
    
    visualize()

    