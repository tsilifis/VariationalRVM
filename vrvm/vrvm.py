import numpy as np
from .kern import RBF

__all__ = ["VRVM"]

class VRVM(object):

	_train_data = None 

	_Phi = None 

	_kernel = None 

	_n_kernels = None 

	_q_params = None 

	_alpha_params = None 

	_xi = None 

	def __init__(self, input_dim, kernel="rbf", lengthscale=0.5):
		"""
		Initializes a Variational Relevance Vector Machine object. 
		Parameters that need fine-tuning: - Kernel lengthscale
										  - Maxit
										  - Tol
		"""

		self._kernel = RBF(input_dim, corr_length=lengthscale)


	def fit(self, X, y, maxit:int=1000):

		self._train_data = {"X": X, "y": y}
		self._Phi = self._kernel.eval(X)
		Phi_ = np.vstack([np.ones((1, self._Phi.shape[0])), self._Phi])
		a = b = 1e-6
		m_hat = np.zeros(self._Phi.shape[1]+1)
		S_hat = np.eye(self._Phi.shape[1]+1)

		err = 1e+6

		a_hat = (a+0.5) * np.ones(self._Phi.shape[1] + 1)
		b_hat = (b+0.5) * np.ones(self._Phi.shape[1] + 1)

		xi_sq = [np.dot(Phi_[:, i][None, :], np.dot(S_hat + np.dot(m_hat[:, None], m_hat[None, :]), Phi_[:, i]))[0] for i in range(Phi_.shape[1])]

		def Lambda(xi):
			return (1/(4*xi)) * np.tanh(xi/2)

		num_it = 0
		while err > 1e-6 and num_it < maxit:

			# Update alpha params 
			a_hat = a + 0.5
			b_hat = b + 0.5 * (np.diag(S_hat) + m_hat**2)

			# Update omega params 
			S_terms = [Lambda(np.sqrt(xi_sq[i]))*np.dot(Phi_[:, i][:, None], Phi_[:, i][None, :]) for i in range(Phi_.shape[1])]

			S_hat_inv = np.diag(a_hat / b_hat) + 2 * sum(S_terms)
			S_hat_new = np.linalg.solve(S_hat_inv, np.eye(S_hat_inv.shape[0]))
			m_hat_new = 0.5 * np.dot(S_hat_new, np.sum((2*y.flatten()-1)*Phi_, axis=1))

			# Update xi's
			xi_sq = [np.dot(Phi_[:, i][None, :], np.dot(S_hat_new + np.dot(m_hat_new[:, None], m_hat_new[None,:]), Phi_[:, i]))[0] for i in range(Phi_.shape[1])]
			err = np.linalg.norm(m_hat_new.flatten() - m_hat.flatten())
			S_hat = S_hat_new.copy()
			m_hat = m_hat_new.copy()
			print("Mean Squared Error:" + str(err))
			num_it += 1

		self._q_params = {"mu": m_hat, "S": S_hat}


	def predict(self, X):
		"""
		Make prediction on the test data X
		"""
		Phi = self._kernel.eval(self._train_data["X"], Y=X)
		Phi_ = np.vstack([np.ones((1, Phi.shape[1])), Phi])
		y = np.dot(Phi_.T, self._q_params["mu"])
		return 1 / (1 + np.exp(-y))




