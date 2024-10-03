import numpy as np
from sklearn import linear_model

class Perceptron(object):
    """ Percepron classifier.
    Parameters
    ---------------------
    eta : float
        Learning rate (between 0.0 and 1.0)
    n_iter : int
        Passes over the training dataset.
    random_state : int
        Random number generator seed for random weight initialization.
    Attributes
    -----------
    w_ : 1d-array
        Weights after fitting.
    errors_ : list
        Number of misclassifications (updates) in each epoch.
    """
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
        self.reg = linear_model.LinearRegression()
        
    
    def fit(self, X, y):
        """
        Fit training data.
        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
        Training vectors, where n_examples is the number of examples and n_features is the number of features.
        y : array-like, shape = [n_examples]
        Target values.
        Returns
        -------
        self : object

        """
        self.reg.fit(X,y)
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01,size=1 + X.shape[1])
        self.errors_ = []
        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                xi = xi.reshape(1,1)
                y_predict = self.reg.predict(xi)
                update = self.eta * (target - y_predict)
                result_update = update * xi
                # self.w_[1:] += update * xi
                result_update = result_update.reshape(1,)
                self.w_[1:] += result_update
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self        
    def net_input(self, X):
        """
        Calculate net input
        """
        return np.dot(X, self.w_[1:]) + self.w_[0]
    def predict(self, X):
        """
        Return class label after unit step
        """
        return np.where(self.net_input(X) >= 0.0, 1, -1)