import os
import sys
import numpy as np
from PIL import Image
import random
from scipy.spatial.distance import pdist, cdist
import pickle
import argparse

parser = argparse.ArgumentParser(description="arg parser")
parser.add_argument('--k', type=int, default=3)
parser.add_argument('--iter', type=int, default=200)
parser.add_argument('--input', type=str, default='image1.png')
parser.add_argument('--output', type=str, default='output')
parser.add_argument('--s', type=float, default=0.001)
parser.add_argument('--c', type=float, default=0.01)
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

def initial_center(k, mode="random"):
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
    # print(centers_idx)
    return centers_idx

def initial_kernel_kmeans(K, data):
    centers_idx = initial_center(K, mode="kmeans++")

    N = len(data)
    cluster = np.zeros(N, dtype=int)
    for n in range(N):
        dist = np.zeros(K)
        for k in range(K):
            dist[k] = data[n,n] + data[centers_idx[k],centers_idx[k]] - 2*data[n,centers_idx[k]]
        cluster[n] = np.argmin(dist)
    return cluster

def construct_sigma_n(kernelj, cluster, cluster_k):
    ker = kernelj.copy()
    mask = np.where(cluster==cluster_k)
    sigma = np.sum(ker[mask])
    return sigma

def construct_sigma_pq(C, K, kernel, cluster):
    pq = np.zeros(K)
    for k in range(K):
        ker = kernel.copy()
        for n in range(len(kernel)):
            if cluster[n]!=k:
                ker[n,:] = 0
                ker[:,n] = 0
        pq[k] = np.sum(ker)/C[k]/C[k]
    return pq

def construct_C(K, cluster):
    C = np.zeros(K, dtype=int)
    for k in range(K):
        indicator = np.where(cluster==k, 1, 0)
        C[k] = np.sum(indicator)
    return C

def clustering(K, kernel, cluster):
    N = len(kernel)
    new_cluster = np.zeros(N, dtype=int)
    C = construct_C(K, cluster)
    pq = construct_sigma_pq(C, K, kernel, cluster)
    for j in range(N):
        dist = np.zeros(K)
        for k in range(K):
            dist[k] += kernel[j,j] + pq[k]
            dist[k] -= 2/C[k] * construct_sigma_n(kernel[j,:], cluster, k)
        new_cluster[j] = np.argmin(dist)

    return new_cluster

def find_centers(K, data, cluster):
    new_centers = []
    for k in range(K):
        mask = cluster==k
        cluster_k = data[mask]
        new_center_k = np.sum(cluster_k, axis=0) / len(cluster_k)
        new_centers.append(new_center_k)
    return new_centers

def save_png(N, K, cluster, iter):
    colors = np.array([[255,0,0],[0,255,0],[0,0,255],[0,215,175],[95,0,135],[255,255,0],[255,175,0]])
    result = np.zeros((100*100, 3))
    for n in range(N):
        result[n,:] = colors[cluster[n],:]
    
    img = result.reshape(100, 100, 3)
    img = Image.fromarray(np.uint8(img))
    img.save(os.path.join(output_dir, '%06d.png'%iter))

def kernel_kmeans(K, kernel, cluster, iter=1000):
    save_png(len(kernel), K, cluster, 0)
    for i in range(1, iter+1):
        print("iter", i)
        new_cluster = clustering(K, kernel, cluster)
        if(np.linalg.norm((new_cluster-cluster), ord=2)<1e-2):
            break
        cluster = new_cluster
        save_png(len(kernel), K, cluster, i)

if __name__=='__main__':
    output_dir = args.output
    if not os.path.exists(output_dir):
        try:
            os.mkdir(output_dir)
        except:
            raise OSError("Can't create destination directory (%s)!" % (output_dir))

    X_color = load_png() #(10000, 3)
    X_spatial, fi_X = RBF_kernel(X_color, args.s, args.c) #spatial, color
    cluster = initial_kernel_kmeans(args.k, fi_X)
    kernel_kmeans(args.k, fi_X, cluster, iter=args.iter)
