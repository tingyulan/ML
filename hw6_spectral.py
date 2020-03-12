import os
import numpy as np
from PIL import Image
import random
from scipy.spatial.distance import pdist, cdist
import pickle
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser(description="arg parser")
parser.add_argument('--k', type=int, default=2)
parser.add_argument('--iter', type=int, default=200)
parser.add_argument('--input', type=str, default='image1.png')
parser.add_argument('--mode', type=int, default=1, help="0:kmeans, 1:unnormalized spectral, 2:normalized spectral")
parser.add_argument('--s', type=float, default=0.0001)
parser.add_argument('--c', type=float, default=0.001)
parser.add_argument('--output', type=str, default='output')
parser.add_argument('--filename', type=str, default='filename')
args = parser.parse_args()

def load_png():
    img = Image.open(args.input)
    img = np.array(img.getdata()) #(10000,3)
    return img

def RBF_kernel(X, gamma_s, gamma_c): #img, spatial, color
    dist_c = cdist(X,X,'sqeuclidean') #(10000,10000)

    seq = np.arange(0,100)
    c_coord = seq
    for i in range (99):
        c_coord = np.hstack((c_coord, seq))
    c_coord = c_coord.reshape(-1,1)
    r_coord = c_coord.reshape(100,100).T.reshape(-1,1)
    X_s = np.hstack((r_coord, c_coord))
    dist_s = cdist(X_s,X_s,'sqeuclidean')

    RBF_s = np.exp( -gamma_s * dist_s)
    RBF_c = np.exp( -gamma_c * dist_c) #(10000,10000)
    k = np.multiply(RBF_s, RBF_c) #(10000,10000)
    
    return X_s, k

def initial_centers(k):
    centers = list(random.sample(range(0,10000), k))
    return centers

def initial_center(k,data, mode="random"):
    if mode=="random":
        centers_idx = list(random.sample(range(0,10000), k))
    elif mode=="kmeans++":
        centers_idx = []
        centers_idx = list(random.sample(range(0,10000), 1))
        found = 1
        while (found<k):
            dist = np.zeros(10000)
            for i in range(10000):
                min_dist = np.Inf
                for f in range(found):
                    tmp = np.linalg.norm(X_spatial[i,:]-X_spatial[centers_idx[f],:])
                    if tmp<min_dist:
                        min_dist = tmp
                dist[i] = min_dist
            dist = dist/np.sum(dist)
            idx = np.random.choice(np.arange(10000), 1, p=dist)
            centers_idx.append(idx[0])
            found += 1
    
    centers = []
    for i in range(k):
        centers.append(data[k,:])
    return centers

def kernel_dist(data, n, k):
    return data[n,n] + data[k,k] - 2*data[n,k]

def clustering(K, data, centers):
    N = len(data)
    cluster = np.zeros(N, dtype=int)
    for n in range(N):
        c = -1
        min_dist = np.Inf
        for k in range(K):
            dist = np.linalg.norm((data[n]-data[k]), ord=2)
            if dist < min_dist:
                c = k
                min_dist = dist
        cluster[n] = c
    return cluster

def find_centers(K, U, cluster, centers):
    new_centers = []
    for k in range(K):
        mask = cluster==k
        cluster_k = U[mask]
        new_center_k = np.sum(cluster_k, axis=0) / len(cluster_k)
        new_centers.append(new_center_k)
    
    return new_centers

def save_png(N, cluster, centers, iter):
    colors = np.array([[255,0,0],[0,255,0],[0,0,255],[95,0,135],[0,215,175],[255,175,0],[255,255,0]])
    result = np.zeros((100*100, 3))
    for n in range(N):
        result[n,:] = colors[cluster[n],:]
    
    
    img = result.reshape(100, 100, 3)
    img = Image.fromarray(np.uint8(img))
    img.save(os.path.join(output_dir, '%06d.png'%iter))

def kmeans(K, data, centers, iter=1000):
    for i in range(iter):
        print("iter ", i)
        new_cluster = clustering(K, data, centers)
        if i != 0:
            if(np.linalg.norm((new_cluster-cluster), ord=2)<1e-2):
                break
        cluster = new_cluster
        save_png(len(data), cluster, centers, i)
        centers = find_centers(K, data, cluster, centers)
    cluster = clustering(K, data, centers)
    save_png(len(data), cluster, centers, iter)
    return cluster

def construct_Laplacian(W):
    D = np.zeros((W.shape))
    L = np.zeros((W.shape))
    for r in range(len(W)):
        for c in range(len(W)):
            D[r,r] += W[r,c]

    L = D-W
    return D, L

def D_minus_half_square_root(D):
    Dsym = np.zeros((D.shape))
    for i in range(len(D)):
        Dsym[i,i] = D[i,i]**-0.5
    
    return Dsym

def normalize_rows(A):
    row, col = A.shape
    sigma = np.sum(A, axis=1)
    print(A.shape, sigma.shape)
    for r in range(row):
        A[r, :] /= sigma[r]
    
    return A

def save_pkl(filename, isNorm=False):
    if isNorm:
        pkl_data = {'X_spatial': X_spatial, \
                    'W': W, \
                    'D': D, \
                    'L': L, \
                    'Dsym': Dsym, \
                    'Lsym': Lsym}
    else:
        pkl_data = {'X_spatial': X_spatial, \
                    'W': W, \
                    'D': D,\
                    'L': L}
    output = open(filename+".pkl", 'wb')
    pickle.dump(pkl_data, output)
    output.close()

def load_pkl(filename, isNorm=False):
    pkl_file = open(filename+".pkl", 'rb')
    pkl_data = pickle.load(pkl_file)
    pkl_file.close()

    X_spatial = pkl_data['X_spatial']
    W = pkl_data['W']
    D = pkl_data['D']
    L = pkl_data['L']
    if isNorm:
        Dsym = pkl_data['Dsym']
        Lsym = pkl_data['Lsym']
        return X_spatial, W, D, L, Dsym, Lsym
    return X_spatial, W, D, L

def show_eigen(K, data, cluster):
    colors = ['b', 'r']
    plt.clf()
    for idx in range(len(data)):
        plt.scatter(data[idx,0], data[idx,1], c= colors[cluster[idx]])
    plt.savefig("eigen_"+args.filename+".png")
    plt.show()

if __name__=='__main__':
    output_dir = args.output
    if not os.path.exists(output_dir):
        try:
            os.mkdir(output_dir)
        except:
            raise OSError("Can't create destination directory (%s)!" % (output_dir))
    
    filename = args.filename
    if args.mode==1: #unnormalized
        print("ratio spectral")
        X_color = load_png() #(10000, 3)

        # ****************** eigen ******************
        X_spatial, W = RBF_kernel(X_color, args.s, args.c)
        D, L = construct_Laplacian(W)
        save_pkl(filename, isNorm=False)
        
        # X_spatial, W, D, L = load_pkl(filename, isNorm=False)
        # *******************************************
        
        # ****************** eigen ******************
        eigenvalue, eigenvector = np.linalg.eig(L)
        eigenvector = eigenvector.T
        np.save(filename+"_eigenvalue.npy", eigenvalue)
        np.save(filename+"_eigenvector.npy", eigenvector)
        # eigenvalue = np.load(filename+"_eigenvalue.npy")
        # eigenvector = np.load(filename+"_eigenvector.npy")
        # *******************************************

        print("here")
        sort_idx = np.argsort(eigenvalue)
        mask = eigenvalue[sort_idx]>0
        sort_idx = sort_idx[mask]

        U = eigenvector[sort_idx[0:args.k]].T
        centers = initial_center(args.k, U, mode="random")
        cluster = kmeans(args.k, U, centers, iter=args.iter)
        np.save(filename+"_cluster.npy", cluster)
#        show_eigen(args.k, U, cluster)
    
    elif args.mode==2:
        print("Normalized Spectral Clustering")
        X_color = load_png() #(10000, 3)

        # ****************** pkl ******************
        X_spatial, W = RBF_kernel(X_color, args.s, args.c)
        D, L = construct_Laplacian(W)
        Dsym = D_minus_half_square_root(D)
        Lsym = Dsym.dot(L).dot(Dsym)
        save_pkl(filename, isNorm=True)

        # X_spatial, W, D, L, Dsym, Lsym = load_pkl(filename, isNorm=True)
        # *******************************************
        
        # ****************** eigen ******************
        eigenvalue, eigenvector = np.linalg.eig(Lsym)
        eigenvector = eigenvector.T
        np.save(filename+"_eigenvalue.npy", eigenvalue)
        np.save(filename+"_eigenvector.npy", eigenvector)
        # eigenvalue = np.load(filename+"_eigenvalue.npy")
        # eigenvector = np.load(filename+"_eigenvector.npy")
        # *******************************************
        sort_idx = np.argsort(eigenvalue)
        mask = eigenvalue[sort_idx]>0
        sort_idx = sort_idx[mask]
        
        U = eigenvector[sort_idx[0:args.k]].T
        T = normalize_rows(U)
        centers = initial_center(args.k, T, mode="random")
        cluster = kmeans(args.k, T, centers, iter=args.iter)
        np.save(filename+"_cluster.npy", cluster)
#        show_eigen(args.k, T, cluster)
        

