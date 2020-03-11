import numpy as np
from libsvm.svmutil import *
from scipy.spatial.distance import *
import time


def read_dataset(filename):
    with open(filename, 'r') as fp:
        lines = fp.readlines()
    lines = np.array([line.strip().split(',')
                      for line in lines], dtype="float64")
    return lines


def read_mnist():
    X_train = read_dataset("X_train.csv")  # (5000,784)
    X_test = read_dataset("X_test.csv")  # (2500,784)
    Y_train = read_dataset("Y_train.csv")  # (5000,1)
    Y_test = read_dataset("Y_test.csv")  # (2500,1)

    Y_train = Y_train[:, 0]
    Y_test = Y_test[:, 0]

    return X_train, X_test, Y_train, Y_test


def svm(kernel_type, param_str, record_time=False):  # -v n: n-fold vross validation mode
    print("\n", kernel_type)
    param_str = param_str + " -q"  # -s svm_type: default=0=C_SVC
    param = svm_parameter(param_str)

    time_start = time.time()
    prob = svm_problem(Y_train, X_train)
    model = svm_train(prob, param)
    # p_acc: accuracy(for classification), mean-squared error, squared correlation coefficient(for regression)
    _, test_acc, _ = svm_predict(
        Y_test, X_test, model)  # p_label, p_acc, p_val
    time_end = time.time()
    if record_time:
        print("Time to classify: %0.2f seconds." % (time_end-time_start))

    return test_acc[0]  # float


# -v n: n-fold vross validation mode
def gridsearch_svm(kernel_type, param_str, fold=0, record_time=False):
    print(kernel_type)
    if fold:
        param_str = param_str + " -v {:d}".format(int(fold))
    param_str = param_str + " -q"  # -s svm_type: default=0=C_SVC
    param = svm_parameter(param_str)

    time_start = time.time()
    prob = svm_problem(Y_train, X_train)
    val_acc = svm_train(prob, param)
    time_end = time.time()
    if record_time:
        print("Time to train: %0.2f seconds." % (time_end-time_start))

    return val_acc


def GridSearch(kernel_type):
    c = [1e-4, 1e-3, 1e-2, 0.1, 1, 10, 100, 1000, 10000]
    d = range(0, 10)
    g = [1/784, 1, 1e-1, 1e-2, 1e-3, 1e-4]
    r = range(-10, 10, 1)
    n = 3

    acc_record = []
    max_acc = 0.0
    if kernel_type == "Linear":
        for ci in range(len(c)):
            param_str = " -c "+str(c[ci])
            acc = gridsearch_svm(kernel_type+param_str,
                                 "-t 0"+param_str, fold=n)
            if acc > max_acc:
                max_acc = acc
                max_param = param_str

    elif kernel_type == "Polynomial":
        for ci in range(len(c)):
            param_str = " -c "+str(c[ci])
            for ri in range(len(r)):
                param_str_r = param_str+" -r "+str(r[ri])
                for gi in range(len(g)):
                    param_str_g = param_str_r+" -g "+str(g[gi])
                    for di in range(len(d)):
                        param_str_d = param_str_g+" -d "+str(d[di])
                        acc = gridsearch_svm(
                            kernel_type+param_str_d, "-t 1"+param_str_d, fold=n)
                        # acc_record.append( acc )
                        if acc > max_acc:
                            max_acc = acc
                            max_param = param_str_d

    elif kernel_type == "RBF":
        for ci in range(len(c)):
            param_str = " -c "+str(c[ci])
            for gi in range(len(g)):
                param_str_g = param_str+" -g "+str(g[gi])
                acc = gridsearch_svm(
                    kernel_type+param_str_g, "-t 2"+param_str_g, fold=n)
                # acc_record.append( acc )
                if acc > max_acc:
                    max_acc = acc
                    max_param = param_str_g

    print("============================")
    print(kernel_type)
    print("Best Parameters:", max_param)
    print("Max accuracy", max_acc)
    svm("Testing"+kernel_type, max_param)
    print("============================")


def linear_kernel(u, v):
    return u.dot(v.T)


def RBF_kernel(u, v, gamma):
    return np.exp(gamma * cdist(u, v, 'sqeuclidean'))


if __name__ == '__main__':
    X_train, X_test, Y_train, Y_test = read_mnist()
    mode = 2
    # ******* part A *******
    if mode == 0:
        svm("Linear", "-t 0")  # -c
        svm("Polynomial", "-t 1")  # -c, -r, -g, -d
        svm("RBF", "-t 2")  # -c, -g

    # ******* part B *******
    if mode == 1:
        GridSearch("Linear")
        GridSearch("Polynomial")
        GridSearch("RBF")

    # ******* part C *******
    if mode == 2:
        g = [1/784, 1, 1e-1, 1e-2, 1e-3, 1e-4]
        c = [1e-4, 1e-3, 1e-2, 0.1, 1, 10, 100, 1000, 10000]

        x = read_dataset("X_train.csv")  # (5000,784)
        row, col = x.shape

        max_acc = 0.0
        g_best = 0
        linear_k = linear_kernel(x, x)
        for gi in range(len(g)):
            rbf_k = RBF_kernel(x, x, -g[gi])
            my_k = linear_k + rbf_k
            my_k = np.hstack((np.arange(1, row+1).reshape(-1, 1), my_k))
            prob = svm_problem(Y_train, my_k, isKernel=True)
            for ci in range(len(c)):
                param_str = "-t 4 -c " + str(c[ci]) + " -v 3 -q"
                param_rec = "-t 4 -c " + str(c[ci]) + " -q"
                print("-g", g[gi], param_str)
                param = svm_parameter(param_str)
                val_acc = svm_train(prob, param)

                if val_acc > max_acc:
                    max_acc = val_acc
                    max_param = param_rec
                    g_best = gi
        print("============================")
        print("Best Parameters:", " -g", g[g_best], max_param)
        print("Max accuracy:", max_acc)

        rbf_k = RBF_kernel(x, x, -g[g_best])
        my_k = linear_k + rbf_k
        my_k = np.hstack((np.arange(1, row+1).reshape(-1, 1), my_k))
        prob = svm_problem(Y_train, my_k, isKernel=True)
        param = svm_parameter(max_param)
        model = svm_train(prob, param)

        x_test = read_dataset("X_test.csv")  # (2500,784)
        row, col = x_test.shape
        linear_k = linear_kernel(x_test, x_test)
        rbf_k = RBF_kernel(x_test, x_test, -g[g_best])
        my_k = linear_k + rbf_k
        my_k = np.hstack((np.arange(1, row+1).reshape(-1, 1), my_k))
        _, test_acc, _ = svm_predict(Y_test, my_k, model)
        print("============================")
