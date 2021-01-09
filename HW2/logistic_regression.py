import pandas as pd
import matplotlib.pyplot as plt
import pprint
import sys
import numpy as np
from argparse import ArgumentParser, ArgumentTypeError, Namespace
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.model_selection import RepeatedStratifiedKFold
from typing import List, Tuple, Dict
from statistics import mean
from scipy.special import expit, expm1
from scipy.linalg import inv
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score


def cv_info_fixer(training_info: pd.DataFrame) -> pd.DataFrame:
    """
    Fix problems in Info training set, e.g. missing values and categorical values etc.
    :param training_info: Info training set
    :return: new pandas DataFrame
    """
    info_log('=== Fix Info training set ===')
    new_info = training_info.copy()

    # Fill NaN with 0
    new_info['Comorbidities'] = new_info['Comorbidities'].fillna(0)
    new_info['Antibiotics'] = new_info['Antibiotics'].fillna(0)
    new_info['Bacteria'] = new_info['Bacteria'].fillna(0)

    # Assign unique values to distinct values in the column
    new_info['Comorbidities'] = new_info.groupby('Comorbidities').ngroup()

    # Assign 0 if it's 0, 1 if it's a string
    new_info['Antibiotics'] = new_info.groupby('Antibiotics').ngroup().astype(bool).astype(int)
    new_info['Bacteria'] = new_info.groupby('Bacteria').ngroup().astype(bool).astype(int)

    return new_info


def cv_tpr_fixer(training_tpr: pd.DataFrame) -> pd.DataFrame:
    """
    Fix problems in TPR training set, e.g. missing values etc.
    :param training_tpr: TPR training set
    :return: new pandas DataFrame
    """
    info_log('=== Fix TPR training set ===')

    # Group all data by patient number
    sectors = training_tpr.groupby('No')

    # Get means of each patient
    patients = sectors.mean()

    # Standardize all columns
    patients[['T', 'P', 'R', 'NBPS', 'NBPD']] = StandardScaler().fit_transform(
        patients[['T', 'P', 'R', 'NBPS', 'NBPD']])

    return patients


def predict_info_fixer(training_info: pd.DataFrame, testing_info: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fix problems in Info training and testing set, e.g. missing values and categorical values etc.
    :param training_info: Info training set
    :param testing_info: Info testing set
    :return: new pandas DataFrame
    """
    info_log('=== Fix Info training and testing set ===')

    new_tr_info, new_ts_info = training_info.copy(), testing_info.copy()
    number_of_tr = len(new_tr_info.index)
    new_info = new_tr_info.append(new_ts_info)

    # Fill NaN with 0
    new_info['Comorbidities'] = new_info['Comorbidities'].fillna(0)
    new_info['Antibiotics'] = new_info['Antibiotics'].fillna(0)
    new_info['Bacteria'] = new_info['Bacteria'].fillna(0)

    # Assign unique values to distinct values in the column
    new_info['Comorbidities'] = new_info.groupby('Comorbidities').ngroup()

    # Assign 0 if it's 0, 1 if it's a string
    new_info['Antibiotics'] = new_info.groupby('Antibiotics').ngroup().astype(bool).astype(int)
    new_info['Bacteria'] = new_info.groupby('Bacteria').ngroup().astype(bool).astype(int)

    # Write them back to training and testing
    new_tr_info[['Comorbidities', 'Antibiotics', 'Bacteria']] = new_info[
                                                                    ['Comorbidities', 'Antibiotics', 'Bacteria']].iloc[
                                                                :number_of_tr]
    new_ts_info[['Comorbidities', 'Antibiotics', 'Bacteria']] = new_info[
                                                                    ['Comorbidities', 'Antibiotics', 'Bacteria']].iloc[
                                                                number_of_tr:]

    return new_tr_info, new_ts_info


def predict_tpr_fixer(training_tpr: pd.DataFrame, testing_tpr: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fix problems in TPR training and testing set, e.g. missing values etc.
    :param training_tpr: TPR training set
    :param testing_tpr: TPR testing set
    :return: new pandas DataFrame
    """
    info_log('=== Fix TPR training and testing set ===')

    # Group all training data by patient number
    sectors = training_tpr.groupby('No')

    # Get means of each patient
    training_patients = sectors.mean()

    # Standardize all columns
    ss = StandardScaler()
    training_patients[['T', 'P', 'R', 'NBPS', 'NBPD']] = ss.fit_transform(
        training_patients[['T', 'P', 'R', 'NBPS', 'NBPD']])

    # Get mean and variance of training data
    m = ss.mean_
    variance = ss.var_

    # Group all testing data by patient number
    sectors = testing_tpr.groupby('No')

    # Get means of each patient
    testing_patients = sectors.mean()

    # Scale all columns
    testing_patients[['T', 'P', 'R', 'NBPS', 'NBPD']] = (testing_patients[
                                                             ['T', 'P', 'R', 'NBPS', 'NBPD']] - m) / np.sqrt(variance)

    return training_patients, testing_patients


def feature_selection(training_data: pd.DataFrame, training_target: pd.Series, k) -> List[str]:
    """
    Compute and show all mutual information between features and target
    :param training_data: training data set
    :param training_target: training target
    :param k: number of required features
    :return: List of feature names
    """
    # Use mutual information to select top k features
    list_of_col = SelectKBest(mutual_info_classif, k=k).fit(training_data, training_target).get_support(indices=True)
    features = list(map(list(training_data).__getitem__, list_of_col))

    return features


def cross_validator(training_data: pd.DataFrame, training_target: pd.Series, learning_rate: float, regularization: int,
                    penalty: float) -> None:
    """
    Use cross validation to test each model
    :param training_data: training data set
    :param training_target: training_target
    :param learning_rate: learning rate
    :param regularization: 0: without L2 penalty, 1: with L2 penalty
    :param penalty: hyperparameter of regularization
    :return: None
    """
    info_log('=== Perform cross validation ===')

    # Setup K fold
    skf = RepeatedStratifiedKFold(n_repeats=10, random_state=0)

    # Setup number of total features
    total_features = len(list(training_data)) + 1

    # Run 10 times to get the average
    iteration = 0
    methods = ['Gradient descent', "Newton's method", 'sklearn']
    accuracy = {name: {f'{i}': [] for i in range(2, total_features)} for name in methods}
    f1 = {name: {f'{i}': [] for i in range(2, total_features)} for name in methods}
    for train_index, test_index in skf.split(training_data, training_target):
        # Get training set and testing set
        data_train, target_train = training_data.iloc[train_index.tolist()], training_target.iloc[
            train_index.tolist()]
        data_test, target_test = training_data.iloc[test_index.tolist()], training_target.iloc[test_index.tolist()]

        # Use training set to select features
        for k in range(5, total_features):
            info_log(f'=== Iteration: {iteration}, Num of features: {k} ===')
            features = feature_selection(data_train, target_train, k)

            # Concatenate training data
            concatenated_train = pd.concat([data_train[features], target_train], axis=1, ignore_index=True)
            concatenated_train.columns = features + ['Target']

            # Concatenate testing data
            concatenated_test = pd.concat([data_test[features], target_test], axis=1, ignore_index=True)
            concatenated_test.columns = features + ['Target']

            # Logistic regression without penalty
            gd_weight, nm_weight = logistic_regression(concatenated_train, len(features), learning_rate, regularization,
                                                       penalty)

            # Classify testing data
            test_result = classify(concatenated_test, gd_weight, nm_weight, len(features))
            accuracy['Gradient descent'][f'{k}'].append(test_result['gd'][0])
            f1['Gradient descent'][f'{k}'].append(test_result['gd'][1])
            accuracy["Newton's method"][f'{k}'].append(test_result['nm'][0])
            f1["Newton's method"][f'{k}'].append(test_result['nm'][1])

            # Use logistic regression to train and predict
            acc, score, params, intercept, prediction = lr(data_train, target_train, data_test, target_test, features)
            accuracy['sklearn'][f'{k}'].append(acc)
            f1['sklearn'][f'{k}'].append(score)

            '''pp.pprint(gd_weight)
            pp.pprint(nm_weight)
            pp.pprint(params)
            pp.pprint(intercept)
            pp.pprint(test_result['gd'][2])
            pp.pprint(test_result['gd'][0])
            pp.pprint(test_result['gd'][1])
            pp.pprint(test_result['nm'][2])
            pp.pprint(test_result['nm'][0])
            pp.pprint(test_result['nm'][1])
            pp.pprint(prediction)
            pp.pprint(acc)
            pp.pprint(score)
            return'''

        iteration += 1

    # Print average results
    fig = plt.figure(1)
    fig.canvas.set_window_title('Average results')

    # Print results and plot
    print('=== Accuracy ===')
    plt.subplot(121)
    plt.title('Accuracy')
    plot_and_print(accuracy)

    print('\n=== F1 score ===')
    plt.subplot(122)
    plt.title('F1 score')
    plot_and_print(f1)

    plt.tight_layout()
    plt.show()


def lr(data_train: pd.DataFrame, target_train: pd.Series, data_test: pd.DataFrame,
       target_test: pd.Series, features: List[str]) -> Tuple[float, float, any, any, any]:
    """
    Logistic regression
    :param data_train: training data set
    :param target_train: training target
    :param data_test: testing data set
    :param target_test: testing target
    :param features: selected features
    :return: accuracy, f1 score, prediction and confusion matrix
    """
    # Train and predict
    nb = LogisticRegression(max_iter=1000, penalty='l2').fit(data_train[features], target_train)
    prediction = nb.predict(data_test[features])

    # Return accuracy, f1 score and prediction
    return accuracy_score(prediction, target_test), f1_score(prediction,
                                                             target_test), nb.coef_, nb.intercept_, prediction


def logistic_regression(training_data: pd.DataFrame, num_of_features: int, learning_rate: float, regularization: int,
                        penalty: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Logistic regression with gradient descent and Newton method
    :param training_data: training data set
    :param num_of_features: number of selected features
    :param learning_rate: learning rate
    :param regularization: 0: without L2 penalty, 1: with L2 penalty
    :param penalty: hyperparameter of regularization
    :return: Weights
    """
    num_of_data = len(training_data)

    # Set up Φ and group
    group = training_data['Target'].to_numpy().reshape((num_of_data, 1))
    phi = np.ones((num_of_data, num_of_features + 1))
    del training_data['Target']
    phi[:, 1:] = training_data.to_numpy()

    # Get gradient descent result
    gd_omega = gradient_descent(phi, group, num_of_features, learning_rate, regularization, penalty)

    # Get Newton method result
    nm_omega = newton_method(phi, group, num_of_data, num_of_features, learning_rate, regularization, penalty)

    return gd_omega, nm_omega


def gradient_descent(phi: np.ndarray, group: np.ndarray, num_of_features: int, learning_rate: float,
                     regularization: int, penalty: float) -> np.ndarray:
    """
    Gradient descent
    :param phi: Φ matrix
    :param group: group of each data point
    :param num_of_features: number of features
    :param learning_rate: learning rate
    :param regularization: 0: without L2 penalty, 1: with L2 penalty
    :param penalty: hyperparameter of regularization
    :return: weight vector omega
    """
    info_log('=== gradient descent ===')

    # Set up initial guess of omega
    omega = np.zeros((num_of_features + 1, 1))

    # Get optimal weight vector omega
    count = 0
    while True:
        count += 1
        old_omega = omega.copy()

        # Update omega
        if regularization:
            # With L2 penalty
            omega -= learning_rate * (get_delta_j(phi, omega, group) - penalty * omega - 0.75 * old_omega) / len(phi)
        else:
            # Without L2 penalty
            omega -= learning_rate * (get_delta_j(phi, omega, group) - 0.75 * old_omega) / len(phi)

        if np.linalg.norm(omega - old_omega) < 1e-7 or count > 5000:
            break

    return omega


def newton_method(phi: np.ndarray, group: np.ndarray, num_of_data: int, num_of_features: int, learning_rate: float,
                  regularization: int, penalty: float) -> np.ndarray:
    """
    Newton method
    :param phi: Φ matrix
    :param group: group of each data point
    :param num_of_data: number of data
    :param num_of_features: number of features
    :param learning_rate: learning rate
    :param regularization: 0: without L2 penalty, 1: with L2 penalty
    :param penalty: hyperparameter of regularization
    :return: weight vector omega
    """
    info_log("=== Newton's method ===")

    # Set up initial guess of omega
    omega = np.zeros((num_of_features + 1, 1))

    # Set up D matrix for hessian matrix
    d = np.zeros((num_of_data, num_of_data))

    # Get optimal weight vector omega
    count = 0
    while True:
        count += 1
        old_omega = omega.copy()

        # Fill in values in the diagonal of D matrix
        product = phi.dot(omega)
        diagonal = (expm1(-product) + 1e-11) * (np.power(expit(product), 2) + 1e-11)
        np.fill_diagonal(d, diagonal)

        # Set up hessian matrix
        hessian = phi.T.dot(d.dot(phi))

        # Update omega
        if regularization:
            # With L2 penalty
            try:
                # Use Newton method
                omega -= learning_rate * (inv(hessian).dot(
                    get_delta_j(phi, omega, group)) - penalty * omega - 0.75 * old_omega) / len(phi)
            except:
                # Use gradient descent if hessian is singular or infinite
                omega -= learning_rate * (
                        get_delta_j(phi, omega, group) - penalty * omega - 0.75 * old_omega) / len(phi)
        else:
            # Without L2 penalty
            try:
                # Use Newton method
                omega -= learning_rate * (inv(hessian).dot(get_delta_j(phi, omega, group)) - 0.75 * old_omega) / len(
                    phi)
            except:
                # Use gradient descent if hessian is singular or infinite
                omega -= learning_rate * (get_delta_j(phi, omega, group) - 0.75 * old_omega) / len(phi)

        if np.linalg.norm(omega - old_omega) < 1e-7 or count > 5000:
            break

    return omega


def get_delta_j(phi: np.ndarray, omega: np.ndarray, group: np.ndarray) -> np.ndarray:
    """
    Compute gradient J
    :param phi: Φ matrix
    :param omega: weight vector omega
    :param group: group of each data point
    :return: gradient J
    """
    return phi.T.dot(expit(phi.dot(omega)) - group)


def classify(testing_data: pd.DataFrame, gd_weight: np.ndarray, nm_weight: np.ndarray, num_of_features: int) -> Dict[
    str, List[float or any]]:
    """
    Plot and print the results in score
    :param testing_data: testing data set
    :param gd_weight: weights from gradient descent
    :param nm_weight: weights from Newton's method
    :param num_of_features: number of features
    :return: accuracy and f1-score of gradient descent and Newton's method
    """
    num_of_data = len(testing_data)

    # Set up Φ and group
    group = testing_data['Target'].to_numpy()
    phi = np.ones((num_of_data, num_of_features + 1))
    del testing_data['Target']
    phi[:, 1:] = testing_data.to_numpy()

    # Get results of gradient descent
    gd_result = expit(phi.dot(gd_weight))
    gd_result[gd_result >= 0.5] = 1
    gd_result[gd_result < 0.5] = 0
    gd_result = gd_result.reshape(num_of_data).astype(int)

    # Get results of Newton's method
    nm_result = expit(phi.dot(nm_weight))
    nm_result[nm_result >= 0.5] = 1
    nm_result[nm_result < 0.5] = 0
    nm_result = nm_result.reshape(num_of_data).astype(int)

    return {'gd': [accuracy_score(gd_result, group), f1_score(gd_result, group), gd_result],
            'nm': [accuracy_score(nm_result, group), f1_score(nm_result, group), nm_result]}


def plot_and_print(score: Dict[str, Dict[str, List[float]]]) -> None:
    """
    Plot and print the results in score
    :param score: Dictionary of every classifier's results
    :return: None
    """
    for method, results in score.items():
        print(f'=== {method} ===')
        mean_values = []
        for num_of_features, values in results.items():
            mean_values.append(mean(values))
            print(f'{num_of_features}: {mean(values)}')
        print()
        plt.plot(list(results.keys()), mean_values, label=f'{method}')
    plt.ylim(0.0, 1.0)
    plt.legend()


def check_regularization_range(value: str) -> float:
    """
    Check whether penalty is positive float
    :param value: string value
    :return: float value
    """
    float_value = float(value)
    if float_value < 0:
        raise ArgumentTypeError(f'"{value}" is an invalid value. It should be positive float.')
    return float_value


def check_int_range(value: str) -> int:
    """
    Check whether value is 0 or 1
    :param value: string value
    :return: integer value
    """
    int_value = int(value)
    if int_value != 0 and int_value != 1:
        raise ArgumentTypeError(f'"{value}" is an invalid value. It should be 0 or 1.')
    return int_value


def info_log(log: str) -> None:
    """
    Print information log
    :param log: log to be displayed
    :return: None
    """
    if verbosity > 0:
        print(f'[\033[96mINFO\033[00m] {log}')
        sys.stdout.flush()


def error_log(log: str) -> None:
    """
    Print error log
    :param log: log to be displayed
    :return: None
    """
    print(f'[\033[91mERROR\033[00m] {log}')
    sys.stdout.flush()


def parse_arguments() -> Namespace:
    """
    Parse all arguments
    :return: arguments
    """
    parser = ArgumentParser(description='Logistic regression')
    parser.add_argument('-l', '--learning_rate', help='Learning rate', default=0.1, type=float)
    parser.add_argument('-r', '--regularization', help='0: without L2 regularization, 1: with L2 regularization',
                        default=0, type=check_int_range)
    parser.add_argument('-p', '--penalty', help='Hyperparameter of regularization', default=1.0,
                        type=check_regularization_range)
    parser.add_argument('-m', '--mode', help='0: cross validation, 1: prediction', default=0, type=check_int_range)
    parser.add_argument('-v', '--verbosity', help='verbosity level (0-1)', default=0, type=check_int_range)

    return parser.parse_args()


def main(arguments: Namespace) -> None:
    """
    Main function
    :param arguments: arguments parsed from the given command
    :return: None
    """
    # Parse arguments
    learning_rate = arguments.learning_rate
    regularization = arguments.regularization
    penalty = arguments.penalty
    mode = arguments.mode

    # Get training Info sheet
    info_log('=== Loading training data ===')
    training_info = pd.read_excel('data/Training data.xlsx', sheet_name='Info',
                                  names=['No', 'Gender', 'Age', 'Comorbidities', 'Antibiotics', 'Bacteria', 'Target'])
    # Get training TPR sheet
    training_tpr = pd.read_excel('data/Training data.xlsx', sheet_name='TPR')

    # With or without regularization
    if regularization:
        info_log(f'=== With regularization penalty {penalty} ===')
    else:
        info_log('=== Without regularization ===')

    # Cross validation or prediction
    if not mode:
        info_log('=== Cross validation ===')

        # Preprocess Info sheet
        training_info = cv_info_fixer(training_info)

        # Preprocess TPR sheet
        training_tpr = cv_tpr_fixer(training_tpr)

        # Merge Info and TPR
        training_data = pd.merge(training_info, training_tpr, on='No')

        # Get training target
        tr_target = training_data['Target'].copy()
        del training_data['Target']
        del training_data['No']

        cross_validator(training_data, tr_target, learning_rate, regularization, penalty)
    else:
        info_log('=== Prediction ===')

        # Get submission
        submission = pd.read_csv('data/Submission.csv')

        # Get testing Info sheet
        info_log('=== Loading testing data ===')
        testing_info = pd.read_excel('data/testing_data.xlsx', sheet_name='Info',
                                     names=['No', 'Gender', 'Age', 'Comorbidities', 'Antibiotics', 'Bacteria'])
        # Merge ts_info and sub
        testing_info = pd.merge(testing_info, submission, on='No')
        # Get testing TPR sheet
        testing_tpr = pd.read_excel('data/testing_data.xlsx', sheet_name='TPR')

        # Preprocess training and testing Info
        training_info, testing_info = predict_info_fixer(training_info, testing_info)

        # Preprocess training and testing TPR
        training_tpr, testing_tpr = predict_tpr_fixer(training_tpr, testing_tpr)

        # Merge Info and TPR
        training_data = pd.merge(training_info, training_tpr, on='No')
        testing_data = pd.merge(testing_info, testing_tpr, on='No')

        # Get training target
        training_target = training_data['Target'].copy()
        del training_data['Target']
        del training_data['No']
        del testing_data['Target']


if __name__ == '__main__':
    """
    Command: python3 logistic_regression.py [-l learning_rate] [-r (0-1)] [-p penalty] [-m (0-1)] [-v (0-1)]
    """
    pd.set_option('display.max_rows', None)
    pp = pprint.PrettyPrinter()

    # Parse arguments
    args = parse_arguments()
    verbosity = args.verbosity

    # Main function
    main(args)
