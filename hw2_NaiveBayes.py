import numpy as np
import os
import struct
import matplotlib.pyplot as plt
import math
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--mode", type=int, default=0, help="descrete:0, continuous:1")
args = parser.parse_args()

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

def calculate_prior(train_num, train_labels):
    prior = np.zeros(class_num, dtype=float)
    for n in range(train_num):
        prior[train_labels[n]] += 1
    
    prior /= train_num  #is now a probability, sum to 1
    return prior

def pseudo_count(likelihood):
    for n in range(class_num):
        for p in range(pixels):
            for b in range(bin_num):
                if likelihood[n,p,b]==0:
                    likelihood[n,p,b] = 0.001
    return likelihood

def discrete_likelihood(num, images, labels, pixels):
    likelihood = np.zeros((class_num, pixels, bin_num), dtype=float)
    for n in range(num):
        for p in range(pixels):
            b = images[n][p]//8 #b stands for which bins
            likelihood[labels[n]][p][b] += 1

    return likelihood  

def divide_marginal(prob):
    prob = np.divide(prob, sum(prob))
    return prob

def discrete_predict(prior):
    error = 0
    wrong_record=[]
    prob_record=[]

    for n in range (test_num):
        prob = np.zeros(10, dtype=float)
        for c in range (class_num):
            prob[c] += np.log(prior[c])
            for p in range (pixels):
                prob[c] += np.log( likelihood[c][p][test_images[n][p]//8] )
    
        prob = divide_marginal(prob)
        prob_record.append(prob)
        predict = np.argmin(prob)
        if (predict != test_labels[n]):
            error += 1
            # wrong_record.append(n)
    error = float(error)/test_num
    return prob_record, error

def show_discrete_imagination(likelihood):
    zero = np.sum(likelihood[:,:,0:16], axis=2)
    one = np.sum(likelihood[:,:,16:32], axis=2)
    
    im = np.zeros((class_num, pixels))
    im = (one >= zero)*1

    print("Imagination of numbers in Bayesian classifier:\n")
    for c in range(class_num):
        print("{}:".format(c))
        for row in range (28):
            for col in range(28):
                print(im[c][row*28+col], end=' ')
            print(" ")

def show_continuous_imagination(mean):
    im = np.zeros((class_num, pixels))
    im = (mean>=128)*1
    
    print("Imagination of numbers in Bayesian classifier:\n")
    for c in range(class_num):
        print("{}:".format(c))
        for row in range (28):
            for col in range(28):
                print(im[c][row*28+col], end=' ')
            print(" ")

def show_result(prob, num=10000):
    for n in range(num):
        print("Postirior (in log scale):")
        for i in range(class_num):
            print("{}: {}".format(i, prob[n][i]))
        print("Prediction: {}, Ans: {}\n".format(np.argmin(prob[n]), test_labels[n]))

def gaussian_likelihood(train_num, train_images, train_labels, pixels, prior):
    mean = np.zeros((class_num, pixels), dtype=float)
    var = np.zeros((class_num, pixels), dtype=float)
    
    for n in range(train_num):
        for p in range(pixels):
            mean[train_labels[n] , p] += train_images[n, p]

    for c in range(class_num):
        mean[c,:] = np.divide( mean[c,:], prior[c]*train_num)
        
    for n in range(train_num):
        for p in range(pixels):
            var[train_labels[n], p] += (train_images[n,p]-mean[train_labels[n],p])**2
            
    for c in range(class_num):
        var[c,:] = np.divide( var[c,:], prior[c]*train_num )

    return mean, var

def gaussian_predict(prior):
    prob_record=[]
    wrongmax_record = []
    max_error = 0

    for n in range(test_num):
        con_prob = np.zeros(10, dtype=float)
        for c in range(class_num):
            con_prob[c] += np.log(prior[c])
            for p in range(pixels):
                if var[c,p] == 0:
                    continue
                con_prob[c] -= np.log( 2.0*math.pi*var[c,p])/2.0
                con_prob[c] -= ((test_images[n,p]-mean[c,p])**2) / (2.0*var[c,p])
        con_prob = divide_marginal(con_prob)
        prob_record.append(con_prob)
        predict = np.argmin(con_prob)
        if (predict != test_labels[n]):
            max_error += 1
            wrongmax_record.append(n)
            
    error = float(max_error)/test_num
    return prob_record, error

if __name__=='__main__':
    mnist_dir = './data/'
    class_num = 10

    train_num, train_images, train_labels, pixels = readmnist(mnist_dir, 'training')
    test_num, test_images, test_labels, _ = readmnist(mnist_dir, 'testing')
    
    prior = calculate_prior(train_num, train_labels)
    np.save("prioir.npy", prior)
    # prior = np.load("prior.npy")

    if args.mode==0: #DISCRETE
        bin_num = 32
        likelihood = discrete_likelihood(train_num, train_images, train_labels, pixels)
        np.save("likelihood.npy", likelihood)
        # likelihood = np.load("likelihood.npy")
        likelihood_div = np.sum(likelihood, axis=2)
        for c in range(class_num):
            for p in range(pixels):
                likelihood[c,p,:] = likelihood[c,p,:]/likelihood_div[c,p]
        likelihood = pseudo_count(likelihood.copy())
        prob, error = discrete_predict(prior)
        np.save("discrete_probability.npy", prob)
        show_result(prob, num=test_num)
        show_discrete_imagination(likelihood)
        print("\nError rate:", error)
    
    else:
        mean, var= gaussian_likelihood(train_num, train_images, train_labels, pixels, prior)
        np.save('mean.npy', mean)
        np.save('var.npy', var)
        prob, error = gaussian_predict(prior)
        np.save("gaussian_probability.npy", prob)
        show_result(prob, num=test_num)
        show_continuous_imagination(mean)
        print("\nError rate:", error)
