import numpy as np 
import scipy.stats as st
import matplotlib.pyplot as plt 
np.random.seed(12345)
import seaborn as sns
sns.set()
import sys
sys.path.insert(0, "../")
from vrvm.vrvm import VRVM, VRVMSparse


def mixture_data(nsamples):
	sigma1 = 5 * np.abs(np.random.normal())
	sigma2 = 5 * np.abs(np.random.normal())
	Sig = np.diag([sigma1, sigma2])
	mu1 = 2 * np.abs(np.random.normal(size=(2,)))
	mu2 = 3 * np.abs(np.random.normal(size=(2,)))
	mu2[1] -= mu2[1]
	mu3 = 6 * np.random.normal(size=(2,))
	mu3[0] += 5
	mu3[1] -= 6
	mu4 = -2 * np.random.normal(size=(2,))

	print(mu1, Sig)
	print(mu2)
	w1 = st.uniform().rvs(size=(nsamples, 1))
	w2 = st.uniform().rvs(size=(nsamples, 1))
	X1 = (w1 > 0.5) * st.multivariate_normal(mu1, Sig).rvs(size=(nsamples)) + (w1 < 0.5) * st.multivariate_normal(mu2, Sig).rvs(size=(nsamples))
	X2 = (w2 > 0.5) * st.multivariate_normal(mu3, Sig).rvs(size=(nsamples)) + (w2 < 0.5) * st.multivariate_normal(mu4, Sig).rvs(size=(nsamples))

	return X1, X2


if __name__ == "__main__":

	N = 50
	X1, X2 = mixture_data(N)
	#plt.scatter(X1[:, 0], X1[:, 1], marker="x")
	#plt.scatter(X2[:, 0], X2[:, 1], marker="o", s=1)
	#plt.show()

	a = np.arange(2*N)
	np.random.shuffle(a)
	X = np.vstack([X1, X2])[list(a), :]
	Y = np.vstack([np.zeros((N, 1)), np.ones((N, 1))])[list(a)]

	from sklearn import svm
	#from sklearn.inspection import DecisionBoundaryDisplay

	clf = svm.SVC()
	clf.fit(X, Y.flatten())

	rvm = VRVM(2, lengthscale=1.)
	rvm.fit(X, Y, maxit=100)
	x1 = np.linspace(-5, 9, 101)
	x2 = np.linspace(-8, 7, 101)
	xx, yy = np.meshgrid(x1, x2)
	X_test = np.hstack([xx.reshape(-1, 1), yy.reshape(-1, 1)])
	plt.plot(rvm._q_params["mu"], "ko", ms=3)
	plt.show()

	sig_pred = rvm.predict(X_test)
	print(sig_pred.reshape(101, 101).shape)
	plt.contourf(x1, x2, (sig_pred.reshape(101, 101)>0.5), 50, alpha=0.4)
	plt.scatter(X[:, 0], X[:, 1], c=Y.flatten(), s=20, edgecolor='k')
	plt.title("VRVM", fontsize=14)
	plt.show()
	#plt.savefig("VRVM.png")

