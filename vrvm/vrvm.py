import numpy as np
from .kern import RBF
import scipy.special as sp
import chaos_basispy as cb
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_score, recall_score


__all__ = ["VRVMBase", "VRVM", "VRVMSparse", "VRVMSoftMax"]


class VRVMBase(object):

    _input_dim = None

    def __init__(self, input_dim):

        self._input_dim = input_dim

    def fit(self, X, y):

        raise NotImplementedError

    def predict_prob(self, X):

        raise NotImplementedError

    
    def cutoff_calibration(self, X_train, Y_train):
        pf_train_mean = self.predict_prob(X_train)
        theta_cutoff = np.linspace(1e-3, 1, 100)

        accuracy_train = np.zeros((np.shape(theta_cutoff)[0], 1))
        f1_score_train = np.zeros((np.share(theta_cutoff)[0], 1))
        precn_pnt = np.zeros((np.shape(theta_cutoff)[0], 1))
        recall_pnt = np.zeros((np.shape(theta_cutoff)[0], 1))
        cnfsn_train = []

        precn_train, recall_train, _ = precision_recall_curve(Y_train, pf_train_mean)

        for i in range(np.shape(theta_cutoff)[0]):
            Y_prediction_train = (pf_train_mean > theta_cutoff[i]).astype(float)
            accuracy_train[i] = accuracy_score(Y_train, Y_prediction_train)
            f1_score_train[i] = f1_score(Y_train, Y_prediction_train)
            cnfsn_train.append(confusion_matrix(Y_train, Y_prediction_train))
            precn_pnt[i] = precision_score(Y_train, Y_prediction_train)
            recall_pnt[i] = recall_score(Y_train, Y_prediction_train)

        indx_best_f1 = np.where(f1_score_train == np.max(f1_score_train))[0]
        indx_best = indx_best_f1[np.where(accuracy_train[indx_best_f1] == np.max(accuracy_train[indx_best_f1]))[0][0]]

        self._cutoff = theta_cutoff[indx_best]
        self.recall_train = recall_train
        self.precn_train = precn_train
        self.recall_pnt = recall_pnt[indx_best]
        self.precn_pnt = precn_pnt[indx_best]
        self.indx_best = indx_best


class VRVM(VRVMBase):

    _train_data = None

    _Phi = None

    _kernel = None

    _n_kernels = None

    _q_params = None

    _alpha_params = None

    _xi = None

    _cutoff = None


    def __init__(self, input_dim, kernel="rbf", lengthscale=0.5):
        """
        Initializes a Variational Relevance Vector Machine object.
        Parameters that need fine-tuning:  - Kernel lengthscale
                                                                           - Maxit
                                                                           - Tol
        """
        super(VRVM, self).__init__(input_dim)

        self._kernel = RBF(input_dim, corr_length=lengthscale)

    def fit(self, X, y, maxit=1000):

        self._train_data = {"X": X, "y": y}
        self._Phi = self._kernel.eval(X)
        Phi_ = np.vstack([np.ones((1, self._Phi.shape[0])), self._Phi])
        a = b = 1e-6
        m_hat = np.zeros(self._Phi.shape[1] + 1)
        S_hat = np.eye(self._Phi.shape[1] + 1)

        err = 1e6

        a_hat = (a + 0.5) * np.ones(self._Phi.shape[1] + 1)
        b_hat = (b + 0.5) * np.ones(self._Phi.shape[1] + 1)

        xi_sq = [
            np.dot(
                Phi_[:, i][None, :],
                np.dot(S_hat + np.dot(m_hat[:, None], m_hat[None, :]), Phi_[:, i]),
            )[0]
            for i in range(Phi_.shape[1])
        ]

        def Lambda(xi):
            return (1 / (4 * xi)) * np.tanh(xi / 2)

        num_it = 0
        while err > 1e-6 and num_it < maxit:

            # Update alpha params
            a_hat = a + 0.5
            b_hat = b + 0.5 * (np.diag(S_hat) + m_hat ** 2)

            # Update omega params
            S_terms = [
                Lambda(np.sqrt(xi_sq[i]))
                * np.dot(Phi_[:, i][:, None], Phi_[:, i][None, :])
                for i in range(Phi_.shape[1])
            ]

            S_hat_inv = np.diag(a_hat / b_hat) + 2 * sum(S_terms)
            S_hat_new = np.linalg.solve(S_hat_inv, np.eye(S_hat_inv.shape[0]))
            m_hat_new = 0.5 * np.dot(
                S_hat_new, np.sum((2 * y.flatten() - 1) * Phi_, axis=1)
            )

            # Update xi's
            xi_sq = [
                np.dot(
                    Phi_[:, i][None, :],
                    np.dot(
                        S_hat_new + np.dot(m_hat_new[:, None], m_hat_new[None, :]),
                        Phi_[:, i],
                    ),
                )[0]
                for i in range(Phi_.shape[1])
            ]
            err = np.linalg.norm(m_hat_new.flatten() - m_hat.flatten())
            S_hat = S_hat_new.copy()
            m_hat = m_hat_new.copy()
            print("Mean Squared Error: " + str(err))
            num_it += 1

        self._q_params = {"mu": m_hat, "S": S_hat}

    def predict_prob(self, X):
        """
        Make prediction on the test data X
        """
        Phi = self._kernel.eval(self._train_data["X"], Y=X)
        Phi_ = np.vstack([np.ones((1, Phi.shape[1])), Phi])
        y = np.dot(Phi_.T, self._q_params["mu"])
        return 1 / (1 + np.exp(-y))

    def predict_prob_sparse(self, f, thres=1e-2):
        Phi_ = f.T 
        w = self._q_params["mu"] * (np.abs(self._q_params["mu"]) > thres)
        y = np.dot(Phi_.T, w)
        return 1 / (1 + np.exp(-y))

    def predict(self, f, sparse=False, intercept=True):
        if sparse:
            probs = self.predict_prob_sparse(f)
        else:
            probs = self.predict_prob(f)
            return (probs > 0.5)#self._cutoff)


class VRVMSparse(VRVMBase):

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
        Parameters that need fine-tuning:  - Kernel lengthscale
                                                                           - Maxit
                                                                           - Tol

        """
        super(VRVMSparse, self).__init__(input_dim)

        self._kernel = RBF(input_dim, corr_length=lengthscale)

    def fit(self, X, y, maxit=1000):

        self._train_data = {"X": X, "y": y}
        self._Phi = self._kernel.eval(X)
        Phi_ = np.vstack([np.ones((1, self._Phi.shape[0])), self._Phi])
        a = b = 1e-6
        c, d = 0.2, 1

        m_hat = np.zeros(self._Phi.shape[1] + 1)
        S_hat = np.eye(self._Phi.shape[1] + 1)

        pi_hat = (c / (c + d)) * np.ones(self._Phi.shape[1] + 1)

        err = 1e6

        a_hat = (a + 0.5) * np.ones(self._Phi.shape[1] + 1)
        b_hat = (b + 0.5) * np.ones(self._Phi.shape[1] + 1)
        c_hat = c + pi_hat
        d_hat = d  # + 1 - pi_hat

        xi_sq = [
            np.dot(
                Phi_[:, i][None, :],
                np.dot(
                    S_hat
                    + np.dot((pi_hat * m_hat)[:, None], (pi_hat * m_hat)[None, :]),
                    Phi_[:, i],
                ),
            )[0]
            for i in range(Phi_.shape[1])
        ]

        def Lambda(xi):
            return (1 / (4 * xi)) * np.tanh(xi / 2)

        num_it = 0
        while err > 1e-6 and num_it < maxit:

            # Update alpha params
            a_hat = a + 0.5
            b_hat = b + 0.5 * (np.diag(S_hat) + m_hat ** 2)

            # Update pi params
            c_hat = c + pi_hat
            d_hat = d  # + 1 - pi_hat

            # Update pi params
            mu_Phi = (m_hat * Phi_.T).T
            mu_pi_Phi = (m_hat * pi_hat * Phi_.T).T

            eta_new = (
                sp.digamma(c_hat)
                - sp.digamma(d_hat)
                + 0.5 * m_hat * np.sum((2 * y.flatten() - 1) * Phi_, axis=1)
            )  # - 2 * (Lambda(np.sqrt(xi_sq)) * np.array([np.sum(np.dot(mu_Phi[:,i][:, None], mu_pi_Phi[:, i][None, :])) for i in range(Phi_.shape[1])])).sum()
            for j in range(eta_new.shape[0]):
                A = np.zeros((Phi_.shape[0], Phi_.shape[0]))
                A[j, :] += pi_hat
                A[:, j] += pi_hat
                A *= S_hat + np.dot(m_hat[:, None], m_hat[None, :])
                eta_new[j] += 2 * sum(
                    [
                        Lambda(np.sqrt(xi_sq[i]))
                        * np.dot(Phi_[:, i][None, :], np.dot(A, Phi_[:, i]))[0]
                        for i in range(Phi_.shape[1])
                    ]
                )
            pi_hat_new = np.exp(eta_new) / (1 + np.exp(eta_new))
            # print(eta_new)
            # print(pi_hat_new)
            # Update omega params
            S_terms = [
                Lambda(np.sqrt(xi_sq[i]))
                * np.dot(
                    (pi_hat_new * Phi_[:, i])[:, None],
                    (pi_hat_new * Phi_[:, i])[None, :],
                )
                for i in range(Phi_.shape[1])
            ]

            S_hat_inv = np.diag(a_hat / b_hat) + 2 * sum(S_terms)
            S_hat_new = np.linalg.solve(S_hat_inv, np.eye(S_hat_inv.shape[0]))
            m_hat_new = 0.5 * np.dot(
                S_hat_new,
                np.sum((2 * y.flatten() - 1) * (pi_hat_new * Phi_.T).T, axis=1),
            )

            # Update xi's
            xi_sq = [
                np.dot(
                    Phi_[:, i][None, :],
                    np.dot(
                        S_hat_new
                        + np.dot(
                            (m_hat_new * pi_hat_new)[:, None],
                            (m_hat_new * pi_hat_new)[None, :],
                        ),
                        Phi_[:, i],
                    ),
                )[0]
                for i in range(Phi_.shape[1])
            ]
            # err = np.linalg.norm(S_hat_new.flatten() - S_hat.flatten()) +
            err = np.linalg.norm(m_hat_new.flatten() - m_hat.flatten())

            S_hat = S_hat_new.copy()
            m_hat = m_hat_new.copy()
            pi_hat = pi_hat_new.copy()
            print("Mean Squared Error: " + str(err))
            num_it += 1

        self._q_params = {"mu": m_hat, "S": S_hat, "z": pi_hat}
        # print("Final weights: ")
        # print(m_hat)

    def predict(self, X):
        """
        Make prediction on the test data X

        """
        Phi = self._kernel.eval(self._train_data["X"], Y=X)
        Phi_ = np.vstack([np.ones((1, Phi.shape[1])), Phi])
        y = np.dot(Phi_.T, self._q_params["mu"] * self._q_params["z"])
        return 1 / (1 + np.exp(-y))


class VRVMSoftMax(VRVMBase):

    _train_data = None

    _Phi = None

    _kernel_type = None

    _kernel = None

    _n_kernels = None

    _nlabels = None

    _q_params = None

    _alpha_params = None

    _xi = None

    _cutoff = None

    def __init__(self, input_dim, nlabels=2, kernel="rbf", lengthscale=0.5, pol_order=4):
        """
        Initializes a Variational Relevance Vector Machine object.
        Parameters that need fine-tuning:  
        - Kernel lengthscale
        - Maxit
        - Tol
        """

        super(VRVMSoftMax, self).__init__(input_dim)

        self._nlabels = nlabels
        if kernel == 'rbf':
            self._kernel_type = kernel
            self._kernel = RBF(input_dim, corr_length=lengthscale)
        elif kernel == 'chaos':
            self._kernel_type = kernel
            self._kernel = cb.PolyBasis(dim=input_dim, degree=pol_order)
        else:
            NotImplementedError('No compatible kernel provided')

    def fit(self, X, y, maxit=1000):

        self._train_data = {"X": X, "y": y}
        if self._kernel_type == 'rbf':
            self._Phi = self._kernel.eval(X)
            Phi_ = np.vstack([np.ones((1, self._Phi.shape[0])), self._Phi])
        if self._kernel_type == 'chaos':
            self._Phi = self._kernel(X)
            Phi_ = self._Phi.T
        a = b = 1e-6
        c, d = 0.2, 1

        m_hat = np.vstack([np.zeros(self._Phi.shape[1] + 1) for _ in range(self._nlabels)]) # K x (N+1)
        S_hat = [np.eye(self._Phi.shape[1] + 1) for _ in range(self._nlabels)]

        err = 1e6

        a_hat = np.vstack([(a + 0.5) * np.ones(self._Phi.shape[1] + 1) for _ in range(self._nlabels)]) # K x (N+1)
        b_hat = np.vstack([(b + 0.5) * np.ones(self._Phi.shape[1] + 1) for _ in range(self._nlabels)]) # K x (N+1)

        xi_all = np.vstack([[
            np.dot(
                Phi_[:, i][None, :],
                np.dot(S_hat[k] + np.dot(m_hat[k, :][:, None], m_hat[k, :][None, :]), Phi_[:, i]),
            )[0]
            for i in range(Phi_.shape[1])
        ] for k in range(self._nlabels)] ) # K x N
        #xi_all = np.vstack([xi_sq for _ in range(self._nlabels)]) # K x N


        def Lambda(xi):
            return (1 / (4 * xi)) * np.tanh(xi / 2)

        gamma = np.array([(0.5*(self._nlabels/2 - 1) + (Lambda(np.sqrt(xi_all[:, i]))*np.dot(m_hat, Phi_[:, i])).sum()) / Lambda(np.sqrt(xi_all[:, i])).sum() for i in range(Phi_.shape[1])])


        num_it = 0
        while err > 1e-6 and num_it < maxit:

            # Update alpha params
            a_hat = a + 0.5
            err_all = []
            for k in range(self._nlabels):
                b_hat[k, :] = b + 0.5 * (np.diag(S_hat[k]) + m_hat[k, :] ** 2)

                # Update omega params
                S_terms_k = [
                    Lambda(np.sqrt(xi_all[k, i]))
                    * np.dot(Phi_[:, i][:, None], Phi_[:, i][None, :])
                    for i in range(Phi_.shape[1])
                ]

                S_hat_inv = np.diag(a_hat / b_hat[k, :]) + 2 * sum(S_terms_k)
                S_hat_new = np.linalg.solve(S_hat_inv, np.eye(S_hat_inv.shape[0]))
                m_hat_new = 0.5 * np.dot(
                    S_hat_new, np.sum((2 * y[:, k] - 1 + 4*gamma*Lambda(np.sqrt(xi_all[k, :]))) * Phi_, axis=1)
                )
                #import pdb 
                #pdb.set_trace()
                # Update xi's
                xi_all[k, :] = np.array([ 
                    np.dot(
                        Phi_[:, i][None, :],
                        np.dot(
                            S_hat_new + np.dot(m_hat_new[:, None], m_hat_new[None, :]),
                            Phi_[:, i],
                        ),
                    )[0] - 2*gamma[i]*(m_hat_new * Phi_[:, i]).sum()
                    for i in range(Phi_.shape[1])
                ]) + gamma**2

                err_k = np.linalg.norm(m_hat_new.flatten() - m_hat[k, :].flatten())
                err_all += [err_k]

                S_hat[k] = S_hat_new.copy()
                m_hat[k, :] = m_hat_new.copy()

            # Update gamma
            gamma = np.array([(0.5*(self._nlabels/2 - 1) + (Lambda(np.sqrt(xi_all[:, i]))*np.dot(m_hat, Phi_[:, i])).sum()) / Lambda(np.sqrt(xi_all[:, i])).sum() for i in range(Phi_.shape[1])])

            err = sum(err_all)
            print("Mean Squared Error: " + str(err))
            num_it += 1

        self._q_params = {"mu": m_hat, "S": S_hat}

    def predict_prob(self, X):
        """
        Make prediction on the test data X
        """
        Phi = self._kernel.eval(self._train_data["X"], Y=X)
        Phi_ = np.vstack([np.ones((1, Phi.shape[1])), Phi]) 
        y = np.dot(self._q_params["mu"], Phi_) # K x N
        return np.exp(y) / np.exp(y).sum(axis=0)

    def predict(self, f):
        probs = self.predict_prob(f)
        return np.argmax(probs, axis=0)
