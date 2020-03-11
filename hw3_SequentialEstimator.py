import numpy as np
import math
import matplotlib.pyplot as plt

def univariate_gaussian(m=3.0, s=5.0):
    #Central Limit Theorem
    X = np.sum(np.random.uniform(0.0, 1.0, 12))-6
    return X * s**0.5 + m

def sequential_estimator(m,s):
    print("Data point source function: N({}, {})\n".format(m, s))
    count = 0

    while 1:
        newValue = univariate_gaussian(m,s)
        if count==0:
            count += 1
            variance = 0
            sampleVariance = 0
            mean = newValue
            M2 = 0
        else:
            count += 1
            delta = newValue - mean
            mean += delta/count
            delta2 = newValue - mean
            M2 += delta*delta2
            variance = M2/2
            sampleVariance = M2/(count-1)

        print("#{}# Add data point: {}".format(count, newValue))
        print("Mean = {}\tVariance = {}".format(mean, sampleVariance)) 

        tolerance = 1e-2
        # print(abs(mean-m), abs(sampleVariance-s))
        if abs(mean-m)<=tolerance and abs(sampleVariance-s)<=tolerance:
            break

if __name__=='__main__':
    #-------------INPUT-----------------
    m=3.0; s=5.0
    #-------------END INPUT-------------

    sequential_estimator(m,s)