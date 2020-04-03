import math
import numpy as np
from itertools import combinations_with_replacement

def polynomial_features(X, degree):
    n_samples, n_features = np.shape(X)
    degree_1 = degree

    def index_combinations():
        combs = [combinations_with_replacement(range(n_features), i) for i in range(0, degree + 1)]
        flat_combs = [i for j in combs for i in j]
        return flat_combs                               # Return all possible combination with length 0 to degree

    combinations = index_combinations()
    n_output_features = len(combinations)
    new = np.empty((n_samples, n_output_features))

    for i, index_combs in enumerate(combinations):
        new[:, i] = np.prod(X[:, index_combs], axis=1)

    return new



class PolynomialRegression(object):

    """Performs a non-linear transformation of the data before fitting the model
    and doing predictions which allows for doing non-linear regression.
    Parameters:
    -----------
    degree: int
        The degree of the polynomial that the independent variable X will be transformed to.
    n_iterations: float
        The number of training iterations the algorithm will tune the weights for.
    learning_rate: float
        The step length that will be used when updating the weights.
    """
    def __init__(self, iterations=1000, learning_rate=0.001):
        # No regularization
        self.regularization = lambda x: 0
        self.regularization.grad = lambda x: 0
        self.iterations=iterations,
        self.learning_rate=learning_rate

    def initialize_weights(self, n_features):
        """ Initialize weights randomly [-1/N, 1/N] """
        min = 1 / math.sqrt(n_features)
        self.w = np.random.uniform(-min, min, (n_features,1))

    def fit(self, X, y):

        # Insert constant ones for bias weights
        X = np.insert(X, 0, 1, axis=1)
        self.training_errors = []
        self.initialize_weights(n_features=X.shape[1])

        # Do gradient descent for n_iterations
        iter = int(''.join(map(str,self.iterations)))
        for i in range(iter):
            y_pred = X.dot(self.w)
            # Calculate l2 loss
            mse = np.mean(0.5 * (y - y_pred)**2 + self.regularization(self.w))
            self.training_errors.append(mse)

            # Gradient of l2 loss w.r.t w
            grad_w = -(X.T).dot((y - y_pred)) + self.regularization.grad(self.w)

            # Update the weights
            self.w -= self.learning_rate * grad_w/X.shape[0]

    def predict(self, X):
        #X = polynomial_features(X, degree=self.degree)

        # Insert constant ones for bias weights
        X = np.insert(X, 0, 1, axis=1)
        y_pred = X.dot(self.w)
        return y_pred

