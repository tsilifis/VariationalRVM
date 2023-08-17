import numpy as np 
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF


def f(x):
 
	f1 = 3 + 0.1 * (x[:, 0] - x[:, 1])**2 - (x[:, 0] + x[:, 1]) / np.sqrt(2)
	f2 = 3 + 0.1 * (x[:, 0] - x[:, 1])**2 + (x[:, 0] + x[:, 1]) / np.sqrt(2)
	f3 = (x[:, 0] - x[:, 1]) + 6 / np.sqrt(2)
	f4 = (x[:, 1] - x[:, 0]) + 6 / np.sqrt(2)

	return np.minimum(np.minimum(np.minimum(f1, f2), f3), f4)

if __name__ == "__main__":

	N_train = 100
	x = np.random.normal(size=(N_train, 2))
	y = f(x)
	#plt.hist(y, density=True, bins=100)
	#plt.show()


	Y = 0*(y>=2) + (y<2) * (y>=1) + 2*(y<1)
	Y_new = np.zeros((Y.shape[0], int(Y.max()+1)))
	#for i in range(Y.shape[0]):
	#	Y_new[i, Y[i].astype('int')] = 1

	print(x.shape, Y.shape)

	kernel = 1.0 * RBF([1.0])
	gpc_rbf_isotropic = GaussianProcessClassifier(kernel=kernel).fit(x, Y)
	kernel = 1.0 * RBF([1.0, 1.0])
	gpc_rbf_anisotropic = GaussianProcessClassifier(kernel=kernel).fit(x, Y)

	# create a mesh to plot in
	x1 = np.linspace(-4, 4, 250)
	xx, yy = np.meshgrid(x1, x1)

	titles = ["Isotropic RBF", "Anisotropic RBF"]

	plt.figure(figsize=(10, 5))
	for i, clf in enumerate((gpc_rbf_isotropic, gpc_rbf_anisotropic)):
	    # Plot the predicted probabilities. For that, we will assign a color to
	    # each point in the mesh [x_min, m_max]x[y_min, y_max].
	    plt.subplot(1, 2, i + 1)

	    Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])

	    # Put the result into a color plot
	    Z = Z.reshape((xx.shape[0], xx.shape[1], 3))
	    plt.imshow(Z, extent=(-4, 4, -4, 4), origin="lower")

	    # Plot also the training points
	    plt.scatter(x[:, 0], x[:, 1], c=np.array(["r", "g", "b"])[Y], edgecolors=(0, 0, 0))
	    plt.xlabel("Sepal length")
	    plt.ylabel("Sepal width")
	    plt.xlim(xx.min(), xx.max())
	    plt.ylim(yy.min(), yy.max())
	    plt.xticks(())
	    plt.yticks(())
	    plt.title(
	        "%s, LML: %.3f" % (titles[i], clf.log_marginal_likelihood(clf.kernel_.theta))
	    )

	plt.tight_layout()
	plt.show()

