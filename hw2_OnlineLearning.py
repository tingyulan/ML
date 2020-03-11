import numpy as np
import math
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--a", "--one", type=int, default=0)
parser.add_argument("--b", "--zero", type=int, default=0)
args = parser.parse_args()


def readfile(filename):
    outcomes = []
    with open (filename, 'r') as fp:
        for line in fp:
            outcomes.append(line.strip())
    return outcomes

def Combination(N, m):
    return math.factorial(N)/math.factorial(N-m)/math.factorial(m) 

def Binomial(m, N, p):
    return Combination(N, m) * p**m * (1-p)**(N-m)

def count(outcome):
    zero=0
    one=0
    for i in range(len(outcome)):
        if outcome[i]=='1':
            one += 1
        else:
            zero += 1
    return one, zero

if __name__ == '__main__':
    file_dir = './data/testfile.txt' 
    outcomes = readfile(file_dir)
    
    prior_a= args.a
    prior_b= args.b

for i in range(len(outcomes)):
    posterior_a, posterior_b = count(outcomes[i])
    likelihood = Binomial(posterior_a, posterior_a+posterior_b, posterior_a/(posterior_a+posterior_b))
    posterior_a += prior_a
    posterior_b += prior_b
    
    print("case", i, ":", outcomes[i])
    print("Likelihood:", likelihood)
    print("Beta prior:\ta =", prior_a, "b =", prior_b)
    print("Beta posterior:\ta =", posterior_a, "b =", posterior_b,"\n")
    
    prior_a = posterior_a
    prior_b = posterior_b
