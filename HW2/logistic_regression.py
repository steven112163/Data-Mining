import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit
from math import floor, ceil


class LogisticRegression(object):
    """
    Logistic regression by gradient regression
    :param learning_rate: float, default = 0.1, learning rate
    :param regularization: int, default = 0, 0: without L2 regularization, 1: with L2 regularization
    :param penalty: float, default = 1.0, hyperparameter of regularization
    """

    def __init__(self, learning_rate: float = 0.1, regularization: int = 0, penalty: float = 1.0):
        """
        Constructor
        :param learning_rate: learning rate
        :param regularization: 0: without L2 regularization, 1: with L2 regularization
        :param penalty: Hyperparameter of regularization
        """
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.penalty = penalty
        self.omega = None
        self.features = []

    def fit(self, training_data: pd.DataFrame) -> np.ndarray:
        """
        Logistic regression with gradient descent
        :param training_data: training data set
        :return: weights
        """
        # Calculate parameters
        num_of_data = len(training_data)

        # Set up Φ and group
        group = training_data['Target'].to_numpy().reshape((num_of_data, 1))
        del training_data['Target']
        self.features = list(training_data)
        num_of_features = len(self.features)
        phi = np.ones((num_of_data, num_of_features + 1))
        phi[:, 1:] = training_data.to_numpy()

        # Get gradient descent result
        self.omega = self.gradient_descent(phi, group, num_of_features)

        return self.omega

    def gradient_descent(self, phi: np.ndarray, group: np.ndarray, num_of_features: int) -> np.ndarray:
        """
        Gradient descent
        :param phi: Φ matrix
        :param group: group of each data point
        :param num_of_features: number of features
        :return: weight vector omega
        """
        # Set up initial guess of omega
        omega = np.zeros((num_of_features + 1, 1))

        # Get optimal weight vector omega
        count = 0
        while True:
            count += 1
            old_omega = omega.copy()

            # Update omega
            if self.regularization:
                # With L2 penalty
                omega -= self.learning_rate * (
                        self.get_delta_j(phi, omega, group) - self.penalty * omega - 0.75 * old_omega) / len(phi)
            else:
                # Without L2 penalty
                omega -= self.learning_rate * (self.get_delta_j(phi, omega, group) - 0.75 * old_omega) / len(phi)

            if np.linalg.norm(omega - old_omega) < 1e-7 or count > 5000:
                break

        return omega

    def get_delta_j(self, phi: np.ndarray, omega: np.ndarray, group: np.ndarray) -> np.ndarray:
        """
        Compute gradient J
        :param phi: Φ matrix
        :param omega: weight vector omega
        :param group: group of each data point
        :return: gradient J
        """
        return phi.T.dot(expit(phi.dot(omega)) - group)

    def predict(self, weight: np.ndarray, test_data: pd.DataFrame) -> np.ndarray:
        """
        Plot and print the results in score
        :param weight: weights from gradient descent
        :param test_data: testing data set
        :return: prediction
        """
        # Calculate parameters
        testing_data = test_data[self.features]
        num_of_data = len(testing_data)
        num_of_features = len(list(testing_data))

        # Set up Φ
        phi = np.ones((num_of_data, num_of_features + 1))
        phi[:, 1:] = testing_data.to_numpy()

        # Get results of gradient descent
        weight = weight.reshape((len(weight), 1))
        result = expit(phi.dot(weight))
        result[result >= 0.5] = 1
        result[result < 0.5] = 0
        result = result.reshape(num_of_data).astype(int)

        return result

    def pred_probability(self, weight: np.ndarray, test_data: pd.DataFrame) -> np.ndarray:
        """
        Plot and print the results in score
        :param weight: weights from gradient descent
        :param test_data: testing data set
        :return: prediction
        """
        # Calculate parameters
        testing_data = test_data[self.features]
        num_of_data = len(testing_data)
        num_of_features = len(list(testing_data))

        # Set up Φ
        phi = np.ones((num_of_data, num_of_features + 1))
        phi[:, 1:] = testing_data.to_numpy()

        # Get results of gradient descent
        result = expit(phi.dot(weight)).reshape(num_of_data)

        return result

    def visualize(self) -> None:
        """
        Visualize weights
        :return: None
        """
        fig = plt.figure(1)
        fig.canvas.set_window_title('Feature Weight')

        ax = fig.add_subplot(1, 1, 1)

        colors = ['g' if value > 0 else 'r' for value in self.omega[1:]]
        y_pos = np.arange(len(self.features))

        ax.barh(y_pos, self.omega[1:], align='center', color=colors)
        ax.set_xlim(floor(min(self.omega[1:])), ceil(max(self.omega[1:])))
        ax.set_yticks(y_pos)
        ax.set_yticklabels(self.features)
        ax.invert_yaxis()
        ax.set_title('Feature Weight')
