import math
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

def univariate_gaussian(m=3.0, s=5.0):
    #Central Limit Theorem
    X = np.sum(np.random.uniform(0.0, 1.0, 12))-6
    return X * s**0.5 + m

def polynomial_basis_linear_model(n, a, w):
    x = np.random.uniform(-1.0, 1.0)
    y = 0.0
    for i in range(n):
        y += w[i] * x**i
    y += univariate_gaussian(0,a)
    return x, y

def create_X(newx):
    X = np.zeros((1,n))
    for i in range(n):
        X[0,i]=newx**i
    return X

def draw(x, y, var, lower_bound, upper_bound):
	plt.plot(x, y, color = 'black')
	plt.plot(x, y+var, color = 'red')
	plt.plot(x, y-var, color = 'red')
	plt.xlim(-2.0, 2.0)
	plt.ylim(lower_bound, upper_bound)

def vizualize(x_record, y_record, mean, cov, ten_mean, ten_cov, fifty_mean, fifty_cov, num_point=40):
	x = np.linspace(-2.0, 2.0, num_point)

	#GROUND TRUTH
	plt.subplot(221)
	func = np.poly1d(np.flip(w))
	y = func(x)
	var = (1/a)
	lower_bound = min(y-var)-10
	upper_bound = max(y+var)+10
	plt.title("Ground truth")
	draw(x, y, var, lower_bound, upper_bound)
	
	#ALL
	plt.subplot(222)
	func = np.poly1d(np.flip(np.reshape(mean, n)))
	y = func(x)
	var = np.zeros((num_point))
	for i in range(num_point):
		X = create_X(x[i])
		var[i] = 1/a +  X.dot(cov.dot(X.T))[0][0]
	plt.title("Predict result")
	plt.scatter(x_record, y_record, s=7.0)
	draw(x, y, var, lower_bound, upper_bound)

	#TEN
	plt.subplot(223)
	func = np.poly1d(np.flip(np.reshape(ten_mean, n)))
	y = func(x)
	var = np.zeros((num_point))
	for i in range(num_point):
		X = create_X(x[i])
		var[i] = 1/a +  X.dot(ten_cov.dot(X.T))[0][0]
	plt.title("After 10 incomes")
	plt.scatter(x_record[0:10], y_record[0:10], s=7.0)
	draw(x, y, var, lower_bound, upper_bound)

	#FIFTY
	plt.subplot(224)
	func = np.poly1d(np.flip(np.reshape(fifty_mean, n)))
	y = func(x)
	var = np.zeros((num_point))
	for i in range(num_point):
		X = create_X(x[i])
		var[i] = 1/a +  X.dot(fifty_cov.dot(X.T))[0][0]
	plt.title("After 50 incomes")
	plt.scatter(x_record[0:50], y_record[0:50], s=7.0)
	draw(x, y, var, lower_bound, upper_bound)

	plt.tight_layout()
	plt.show()

def print_online_learning(new_x, new_y, post_mean, post_cov, marginalize_mean, marginalize_cov, count):
	print("Add data point({}, {})".format(new_x, new_y))

	print("Posterior mean:")
	for i in range(n):
		print("  ", format(post_mean[i,0], '0.10f'))

	print("Posterior Variance:")
	for r in range(n):
		print("  ", end=" ")
		for c in range(n):
			if c == n-1:
				print( format(post_cov[r,c], '0.10f') , end=" ")
			else:
				print( format(post_cov[r,c], '0.10f') , end=", ")
			# print("{},".format(post_cov[r,c]), end=" ")
		print(" ")
	
	print("#{}# Predictive distribution ~ N({}, {})".format(count, marginalize_mean, marginalize_cov))
	print("-------------------------------------------")


def baysian_linear_regression(a):
	count = 0
	prior_mean=np.zeros((1,n))
	x_record = []
	y_record = []

	while 1:
		new_x, new_y = polynomial_basis_linear_model(n, a_save, w)
		x_record.append(new_x)
		y_record.append(new_y)
		X = create_X(new_x)

		#UPDATE
		if count == 0:
			bI = b * np.eye(n)
			post_cov = a * X.T.dot(X) + bI
			post_mean = a * np.linalg.inv(post_cov).dot(X.T) * new_y
		else:
			post_cov = a * X.T.dot(X) + prior_cov
			post_mean = np.linalg.inv(post_cov).dot(a * X.T.dot(new_y) + prior_cov.dot(prior_mean))

		#RECORD 10 points and 50 points
		if count == 9:
				ten_mean = post_mean.copy()
				ten_cov = np.linalg.inv(post_cov).copy()
		if count == 49:
				fifty_mean = post_mean.copy()
				fifty_cov = np.linalg.inv(post_cov).copy()
		
		marginalize_mean = np.dot(X,post_mean)
		marginalize_cov = ((1/a) + X.dot( np.linalg.inv(post_cov).dot(X.T) ))

		print_online_learning(new_x, new_y, post_mean, post_cov, marginalize_mean[0,0], marginalize_cov[0,0], count)
		
		if np.linalg.norm(prior_mean-post_mean, ord=2)<1e-3 and count>500:
			break
		#UPDATE PRIOR
		count += 1
		prior_cov = post_cov
		prior_mean = post_mean

	return x_record, y_record, post_mean, np.linalg.inv(post_cov), ten_mean, ten_cov, fifty_mean, fifty_cov

if __name__=='__main__':
	#----------INPUT--------------
	b=1; n=3; a=3
	w = np.array([1.0, 2.0, 3.0])
	#----------END INPUT----------
	a_save = a
	a=1/a

	x_record, y_record, post_mean, post_cov_inv, ten_mean, ten_cov, fifty_mean, fifty_cov = baysian_linear_regression(a)
	vizualize(x_record, y_record, post_mean, post_cov_inv, ten_mean, ten_cov, fifty_mean, fifty_cov)
