import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
import sys
sys.path.insert(0, "../")
from vrvm.vrvm import VRVMSoftMax
sns.set()


def f(x):
 
	f1 = 3 + 0.1 * (x[:, 0] - x[:, 1])**2 - (x[:, 0] + x[:, 1]) / np.sqrt(2)
	f2 = 3 + 0.1 * (x[:, 0] - x[:, 1])**2 + (x[:, 0] + x[:, 1]) / np.sqrt(2)
	f3 = (x[:, 0] - x[:, 1]) + 6 / np.sqrt(2)
	f4 = (x[:, 1] - x[:, 0]) + 6 / np.sqrt(2)

	return np.minimum(np.minimum(np.minimum(f1, f2), f3), f4)


if __name__ == "__main__":

	"""
	x1 = np.linspace(-4, 4, 500)
	xx, yy = np.meshgrid(x1, x1)
	
	y_grid = f(np.hstack([xx.reshape(-1, 1), yy.reshape(-1, 1)]))
	y_grid_c = 0 * (y_grid>=2) + (y_grid<2) * (y_grid>=1) + 2 * (y_grid < 1)
	fig = plt.figure(figsize=(7, 7))
	ax = fig.add_subplot(111)
	ax.contour(x1, x1, y_grid.reshape(500, 500), 50)
	ax.set_xlabel(r'$x_1$', fontsize=18)
	ax.set_ylabel(r'$x_2$', fontsize=18)
	ax.set_title(r'$f(x_1, x_2)$', fontsize=18)
	plt.savefig('fourbranch_results/fx.png')
	plt.show()

	fig = plt.figure(figsize=(7, 7))
	ax = fig.add_subplot(111)
	ax.contourf(x1, x1, y_grid_c.reshape(500, 500), corner_mask=False)#, cmap='viridis')
	ax.set_xlabel(r'$x_1$', fontsize=18)
	ax.set_ylabel(r'$x_2$', fontsize=18)
	ax.set_title(r'$f(x_1, x_2)$', fontsize=18)
	plt.savefig('fourbranch_results/fx_classes.png')
	plt.show()
	"""

	N_train = 500
	x = np.random.normal(size=(N_train, 2))
	y = f(x)
	#plt.hist(y, density=True, bins=100)
	#plt.show()


	Y = 0*(y>=2) + (y<2) * (y>=1) + 2*(y<1)
	Y_new = np.zeros((Y.shape[0], int(Y.max()+1)))
	for i in range(Y.shape[0]):
		Y_new[i, Y[i].astype('int')] = 1

	rvm = VRVMSoftMax(2, nlabels=3, pol_order=3.)
	rvm.fit(x, Y_new, maxit=1000)

	x_test = np.random.normal(size=(10000, 2))
	y_test = f(x_test)
	y_pred = rvm.predict(x_test)

	ind0 = np.where(y_test>=2)[0]
	ind1 = np.where((y_test>=1)&(y_test<2))[0]
	ind2 = np.where(y_test<1)[0]

	ind0_pred = np.where(y_pred==0)[0]
	ind1_pred = np.where(y_pred==1)[0]
	ind2_pred = np.where(y_pred==2)[0]
	#print(ind0)
	fig = plt.figure(figsize=(6,6))
	#ax1 = fig.add_subplot(121)
	#ax.plot(x[:, 0], x[:, 1], 'o', ms=3, color='w', mec='k')
	ax1 = fig.add_subplot(111)
	ax1.plot(x_test[ind0,0], x_test[ind0, 1], 'o', ms=2, color='r', mec='r')
	ax1.plot(x_test[ind1,0], x_test[ind1, 1], 'o', ms=2, color='g', mec='g')
	ax1.plot(x_test[ind2,0], x_test[ind2, 1], 'o', ms=2, color='m', mec='m')
	#plt.savefig('fourbranch_results/f_test.png')
	plt.show()

	fig = plt.figure(figsize=(6,6))
	ax2 = fig.add_subplot(111)
	ax2.plot(x_test[ind0_pred, 0], x_test[ind0_pred, 1], 'o', ms=2, color='r', mec='r')
	ax2.plot(x_test[ind1_pred, 0], x_test[ind1_pred, 1], 'o', ms=2, color='g', mec='g')
	ax2.plot(x_test[ind2_pred, 0], x_test[ind2_pred, 1], 'o', ms=2, color='m', mec='m')
	plt.savefig('fourbranch_results/vrvm_test_500.png')
	plt.show()


