import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import random
from scipy.spatial.distance import cdist

def load_faces(split="Training", root="./Yale_Face_Database"):
    root = os.path.join(root, split)
    files = os.listdir(root)
    num_file = len(files) # 135
    label = np.zeros(num_file, dtype=int)
    faces = np.zeros((num_file, 97*115))
    for idx, f in enumerate(files):
        path = os.path.join(root, f)
        im = Image.open(path) # 195*213=45045
        im = im.resize((97,115))
        im = np.asarray(im)
        row, col = im.shape
        im = im.reshape(1,-1)
        faces[idx,:] = im

        label[idx] = int(files[idx][7:9])
        
    return faces.T, num_file, row, col, files, label # 231, 195

def show_eigenfaces(eigenfaces):
    plt.figure()
    for idx in range(25):
        plt.subplot(5, 5, idx+1)
        plt.axis('off')
        plt.imshow(eigenfaces[idx,:,:], cmap='gray')
    plt.show()

def build_recon_face(eigenfaces, coeff, i):
    result = np.zeros((row, col))
    for eig in range(25):
        result += eigenfaces[eig,:,:]*coeff[i,eig]
    return result

def show_reconstruction_faces(test_faces, test_num_file, eigenfaces):
    coeff = np.zeros((10,25))
    chosen = random.sample(range(test_num_file), 10)
    for i in range(10):
        for eig in range(25):
            coeff[i,eig] = ( test_faces[:,chosen[i]].reshape(1,-1) ).dot( eigenfaces[eig,:,:].reshape(-1,1) )
    
    plt.clf()
    for idx in range(10):
        plt.subplot(2, 10, (idx*2)+1)
        plt.axis('off')
        plt.imshow(test_faces[:,chosen[idx]].reshape(row,col), cmap='gray')

        plt.subplot(2, 10, (idx*2)+2)
        plt.axis('off')
        recon_face = build_recon_face(eigenfaces, coeff, idx)
        plt.imshow(recon_face, cmap='gray')
    plt.show()

def decompose_faces(eigenfaces, faces, num):
    coeff = np.zeros((num, 25))
    for i in range(num):
        for eig in range(25):
            coeff[i,eig] = ( faces[:,i].reshape(1,-1) ).dot( eigenfaces[eig,:,:].reshape(-1,1) )
    return coeff

def classify(test_coeff, test_filename, test_num, test_label, train_coeff, train_filename, train_num, train_label):
    k = 5
    predict = np.zeros(test_num, dtype=int)
    error = 0
    
    dist = np.zeros(train_num)
    for i in range(test_num):
        for j in range(train_num):
            dist[j] = np.linalg.norm(test_coeff[i]-train_coeff[j])
        min_dist = np.argsort(dist)[:k]
        k_predict = train_label[min_dist]
        predict[i] = np.argmax(np.bincount(k_predict))
        if test_label[i]!=predict[i]:
            error += 1
    print("error num:", error, "error rate:", error/test_num)

def kernel(X, mode="RBF"):
    if mode=="linear":
        K = X.dot(X.T)
        
    elif mode=="RBF":
        gamma = 0.0001
        dist = cdist(X,X,'sqeuclidean')
        K = np.exp( -gamma * dist)
    
    return K
        
if __name__ == "__main__":
    faces, num_file, row, col, train_filename, train_label = load_faces() #(45045,num_file=135)
    test_faces, test_num_file, _, _ , test_filename, test_label= load_faces(split="Testing") #(45045, test_num_file=30)

    K = kernel(faces, type="RBF")
    print(K.shape)
    M = row*col
    MM = np.zeros((M, M))/M
    cenK = K - MM.dot(K) - K.dot(MM) + MM.dot(K).dot(MM)

    eigenvalue, eigenvector = np.linalg.eig(K)
    np.save("kernelpca3_K.npy", K)
    np.save("kernelpca3_eigenvalue.npy", eigenvalue)
    np.save("kernelpca3_eigenvector.npy", eigenvector)
    # eigenvalue = np.load("eigenvalue_4.npy")
    # eigenvector = np.load("eigenvector_4.npy")
    # S = np.load("S_4.npy")
    #---------------------------------------------------------

    sort_idx = np.argsort(eigenvalue)[::-1]
    selected_eigenvector = eigenvector[:,sort_idx[0:25]].real
    eigenfaces = selected_eigenvector.reshape(row, col, 25)
    eigenfaces = np.moveaxis(eigenfaces, -1, 0)
    show_eigenfaces(eigenfaces)
    show_reconstruction_faces(test_faces, test_num_file, eigenfaces)

    train_coeff = decompose_faces(eigenfaces, faces, num_file)
    test_coeff = decompose_faces(eigenfaces, test_faces, test_num_file)
    classify(test_coeff, test_filename, test_num_file, test_label, train_coeff, train_filename, num_file, train_label)


