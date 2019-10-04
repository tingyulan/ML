import os
import numpy as np
import random
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--n', type=int, required=True)
parser.add_argument('--lamb', type=float, required=True)
parser.add_argument('--filename', type=str, default='testfile.txt')
args = parser.parse_args()

def readMatrix(A, a, b, line):
    split_line = line.strip().split(',')
    split_line = list(map(float, split_line))
    tmp = []
    for i in range(n):
        tmp.append(split_line[0]**(n-1-i))
    A.append(tmp)
    a.append(split_line[0])
    b.append(split_line[-1])
    return A, a, b

def dot(A, B):
    rowA = len(A)
    colA = len(A[0])
    rowB = len(B)
    colB = len(B[0])
    assert colA==rowB

    C = np.zeros((rowA, colB))
    for r in range(rowA):
        for c in range(colB):
            for k in range(colA):
                C[r,c] += A[r,k]*B[k,c]
    return C

def transpose(M):
    row = len(M)
    col = len(M[0])

    Mt = np.zeros((col,row))
    for r in range(row):
        for c in range(col):
            Mt[c][r] = M[r][c]
    return Mt

def LUdecomposition(Ｍ):
    row = len(Ｍ)
    col = len(Ｍ[0])
    assert row==col #AtA IS ALWAYS A SQUARE MATRIX

    #INITIAL L U MATRIX
    L = np.zeros((row,col))
    U = np.zeros((row,col))
    for c in range(col):
        for r in range(row):
            U[r][c] = M[r][c]
    #L WITH DIAGONAL=1
    for c in range(col):
        L[c][c] = 1

    #START DOING LU DECOMPOSITION
    for c in range(col):
        for r in range(c+1, row):
            L[r][c] = U[r][c]/U[c][c]
            for k in range(c,col):
                U[r][k] = U[r][k] - L[r][c]*U[c][k]

    return L, U

def addlambdaI(M):
    row = len(Ｍ)
    col = len(Ｍ[0])
    assert row==col #AtA IS ALWAYS A SQUARE MATRIX

    for i in range(row):
        M[i][i] += lamb
    
    return M

def LUinverse(L, U):
    row = len(L)
    col = len(L[0])
    assert row==col
    
    I = np.zeros((row, col))
    for r in range(row):
        I[r,r] = 1.0

    #L*L_inv = I
    L_inv = np.zeros((row, col))
    L_inv[0,0] = 1.0/L[0,0] #element 
    for i in range(0, col): #L_inverse back
        for k in range(1,row): #L_origin front
            tmp=0
            for j in range(k-1,-1,-1): #inside elements
                tmp += L[k,j]*L_inv[j,i]
            L_inv[k,i] = (I[k,i]-tmp)/L[k,k]

    #U*A_inv=L_inv
    A_inv = np.zeros((row,col))
    for c in range(col):  #inverse
        A_inv[row-1, c] = L_inv[row-1, c]/U[row-1, row-1]
        for k in range(row-1, -1, -1): #U
            tmp = 0
            for j in range(k+1, row): #inside elements
                tmp += U[k,j]*A_inv[j,c]
            A_inv[k,c] = (L_inv[k,c]-tmp)/U[k,k]

    # print(U_inv==np.linalg.inv(U))
    return A_inv

def calculate_error(x, A, b):
    error = dot(A,transpose([x]))- b
    error = dot(transpose(error),error)
    return error

def initial_readfile(filename):
    A = []; a = []; b = []
    
    fp = open(filename, "r")
    line = fp.readline()
    A, a, b = readMatrix(A, a, b, line)
    while line:
        line = fp.readline()
        if line=="":
            break
        A, a, b = readMatrix(A, a, b, line)
    fp.close()

    A = np.asarray(A)
    a = np.asarray(a)
    b = np.asarray(b)
    b = transpose([b])
    return A, a, b

def show_result(x, error):
    print("Fitting line:", end=' ')
    for i in range(len(x)):
        if(i!=0):
            if(x[i]>=0):
                print("+", end=' ')
            else:
                print("-", end=' ')
                x[i] *= (-1)
        if(i!=len(x)-1):
            print("%.12fx^%d"%(x[i],len(x)-i-1), end=' ')
        else:
            print("%.12f"%(x[i]))
    print("Total error: %.12f"%error)

def LSE(A, b):
    AtA = dot(transpose(A), A)
    AtA = addlambdaI(AtA)
    Atb = dot(transpose(A),b)
    L,U = LUdecomposition(AtA)
    UL_inv = LUinverse(L, U)
    x = dot( UL_inv ,Atb)
    error = calculate_error(x, A, b)
    print("LSE:")
    show_result(x, error)
    return x

def Newton(A, b, iter=100):
    Hessian = transpose(A).dot(A)*2
    L,U = LUdecomposition(Hessian)
    Hessian_inv = LUinverse(L,U)
    gradient_part2 = 2* dot(transpose(A),b)
    
    #INITIAL x
    x = np.random.rand(n,1)

    for _ in range(iter):
        gradient = dot(Hessian,x) - gradient_part2
        step = dot(Hessian_inv,gradient)
        x = x-step

    error = calculate_error(x, A, b)
    print("Newton's Mwthod:")
    show_result(x, error)
    return x


def visualize(a, b, x_LSE, x_Newton):
    mina = min(a)
    maxa = max(a)
    t = np.arange(mina-2, maxa+1+2)
    s_LSE=0
    s_Newton=0
    for i in range(n):
        s_LSE += x_LSE[i]*(t**(n-i-1))
        s_Newton += x_Newton[i]*(t**(n-i-1))

    plt.figure(1)
    plt.subplot(211)
    plt.scatter(a,b, c='r', edgecolors='k')
    plt.plot(t,s_LSE, c='k')
    
    plt.subplot(212)
    plt.scatter(a,b, c='r', edgecolors='k')
    plt.plot(t,s_Newton, c='k')
    
    plt.show()

if __name__ == '__main__':
    # GLOBAL VARIABLE
    n = args.n
    lamb = args.lamb
    filename = args.filename
    #------------------------

    A, a, b = initial_readfile(filename)
    x_LSE = LSE(A, b)
    x_Newton = Newton(A, b, iter=10)
    visualize(a, b, x_LSE, x_Newton)

