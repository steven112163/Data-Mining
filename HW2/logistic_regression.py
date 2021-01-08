import pandas as pd
import matplotlib.pyplot as plt
import pprint
import argparse
import sys
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from typing import List, Tuple, Dict
from statistics import mean
from scipy.special import expit, expm1
from scipy.linalg import inv


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


def cross_validator(training_data: pd.DataFrame, training_target: pd.Series) -> None:
    """
    Use cross validation to test each model
    :param training_data: training data set
    :param training_target: training_target
    :return: None
    """
    info_log('=== Perform cross validation ===')

    # Setup K fold
    skf = RepeatedStratifiedKFold(n_repeats=1, random_state=0)

    # Setup number of total features
    # total_features = len(list(training_data)) + 1
    total_features = 3

    # Run 10 times to get the average
    iteration = 0
    accuracy = {'gd': [], 'nm': []}
    f1 = {'gd': [], 'nm': []}
    for train_index, test_index in skf.split(training_data, training_target):
        # Get training set and testing set
        data_train, target_train = training_data.iloc[train_index.tolist()], training_target.iloc[
            train_index.tolist()]
        data_test, target_test = training_data.iloc[test_index.tolist()], training_target.iloc[test_index.tolist()]

        # Use training set to select features
        for k in range(2, total_features):
            info_log(f'=== Iteration: {iteration}, Num of features: {k} ===')
            features = feature_selection(data_train, target_train, k)

            # Concatenate training data
            concatenated_train = pd.concat([data_train[features], target_train], axis=1, ignore_index=True)
            concatenated_train.columns = features + ['Target']

            # Logistic regression
            gd_weight, nm_weight = logistic_regression(concatenated_train, len(features))

            # Concatenate testing data
            concatenated_test = pd.concat([data_test[features], target_test], axis=1, ignore_index=True)
            concatenated_test.columns = features + ['Target']

            # Classify testing data
            test_result = classify(concatenated_test, gd_weight, nm_weight, len(features))
            accuracy['gd'].append(test_result['gd'][0])
            f1['gd'].append(test_result['gd'][1])
            accuracy['nm'].append(test_result['nm'][0])
            f1['nm'].append(test_result['nm'][1])

        iteration += 1

    # Print average result
    print('Gradient descent:')
    print(f'\tAverage accuracy: {mean(accuracy["gd"])}')
    print(f'\tAverage f1-score: {mean(f1["gd"])}')
    print("Newton's method:")
    print(f'\tAverage accuracy: {mean(accuracy["nm"])}')
    print(f'\tAverage f1-score: {mean(f1["nm"])}')


def logistic_regression(training_data: pd.DataFrame, num_of_features: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Logistic regression with gradient descent and Newton method
    :param training_data: training data set
    :param num_of_features: number of selected features
    :return: Weights
    """
    num_of_data = len(training_data)

    # Set up Φ
    d1 = training_data[training_data['Target'] == 0]
    d2 = training_data[training_data['Target'] == 1]
    del d1['Target']
    del d2['Target']
    phi = np.ones((num_of_data, num_of_features + 1))
    phi[:len(d1), :num_of_features] = d1
    phi[len(d1):, :num_of_features] = d2

    # Set up group number for each data
    group = np.zeros((num_of_data, 1), dtype=int)
    group[len(d1):, 0] = 1

    # Get gradient descent result
    gd_omega = gradient_descent(phi, group, num_of_features)

    # Get Newton method result
    nm_omega = newton_method(phi, group, num_of_data, num_of_features)

    return gd_omega, nm_omega


def gradient_descent(phi: np.ndarray, group: np.ndarray, num_of_features: int) -> np.ndarray:
    """
    Gradient descent
    :param phi: Φ matrix
    :param group: group of each data point
    :param num_of_features: number of features
    :return: weight vector omega
    """
    info_log('=== gradient descent ===')

    # Set up initial guess of omega
    omega = np.random.rand(num_of_features + 1, 1)

    # Get optimal weight vector omega
    count = 0
    while True:
        count += 1
        old_omega = omega.copy()

        # Update omega
        omega += get_delta_j(phi, omega, group)

        if np.linalg.norm(omega - old_omega) < 0.0001 or count > 1000:
            break

    return omega


def newton_method(phi: np.ndarray, group: np.ndarray, num_of_data: int, num_of_features: int) -> np.ndarray:
    """
    Newton method
    :param phi: Φ matrix
    :param group: group of each data point
    :param num_of_data: number of data
    :param num_of_features: number of features
    :return: weight vector omega
    """
    info_log("== Newton's method ==")

    # Set up initial guess of omega
    omega = np.random.rand(num_of_features + 1, 1)

    # Set up D matrix for hessian matrix
    d = np.zeros((num_of_data, num_of_data))

    # Get optimal weight vector omega
    count = 0
    while True:
        count += 1
        old_omega = omega.copy()

        # Fill in values in the diagonal of D matrix
        product = phi.dot(omega)
        diagonal = (expm1(-product) + 1) * np.power(expit(product), 2)
        np.fill_diagonal(d, diagonal)

        # Set up hessian matrix
        hessian = phi.T.dot(d.dot(phi))

        # Update omega
        try:
            # Use Newton method
            omega += inv(hessian).dot(get_delta_j(phi, omega, group))
        except:
            # Use gradient descent if hessian is singular or infinite
            omega += get_delta_j(phi, omega, group)

        if np.linalg.norm(omega - old_omega) < 0.0001 or count > 1000:
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
    return phi.T.dot(group - expit(phi.dot(omega)))


def classify(testing_data: pd.DataFrame, gd_weight: np.ndarray, nm_weight: np.ndarray, num_of_features: int) -> Dict[
    str, List[float]]:
    """
    Plot and print the results in score
    :param testing_data: testing data set
    :param gd_weight: weights from gradient descent
    :param nm_weight: weights from Newton's method
    :param num_of_features: number of features
    :return: accuracy and f1-score of gradient descent and Newton's method
    """
    num_of_data = len(testing_data)

    # Set up Φ
    d1 = testing_data[testing_data['Target'] == 0]
    d2 = testing_data[testing_data['Target'] == 1]
    del d1['Target']
    del d2['Target']
    phi = np.ones((num_of_data, num_of_features + 1))
    phi[:len(d1), :num_of_features] = d1
    phi[len(d1):, :num_of_features] = d2

    # Set up group number for each data
    group = np.zeros((num_of_data, 1), dtype=int)
    group[len(d1):, 0] = 1

    # Get confusion matrix and classification result of gradient descent
    gd_confusion = {'TP': 0, 'FP': 0, 'TN': 0, 'FN': 0}
    for idx in range(num_of_data):
        if phi[idx].dot(gd_weight) >= 0:
            # Class D2
            if group[idx, 0] == 1:
                gd_confusion['TP'] += 1
            else:
                gd_confusion['FP'] += 1
        else:
            # Class D1
            if group[idx, 0] == 0:
                gd_confusion['TN'] += 1
            else:
                gd_confusion['FN'] += 1

    # Get confusion matrix and classification result of Newton's method
    nm_confusion = {'TP': 0, 'FP': 0, 'TN': 0, 'FN': 0}
    for idx in range(num_of_data):
        if phi[idx].dot(nm_weight) >= 0:
            # Class D2
            if group[idx, 0] == 1:
                nm_confusion['TP'] += 1
            else:
                nm_confusion['FP'] += 1
        else:
            # Class D1
            if group[idx, 0] == 0:
                nm_confusion['TN'] += 1
            else:
                nm_confusion['FN'] += 1

    return {'gd': [float(gd_confusion['TP'] + gd_confusion['TN']) / num_of_data,
                   gd_confusion['TP'] / (gd_confusion['TP'] + 0.5 * gd_confusion['FP'] + 0.5 * gd_confusion['FN'])],
            'nm': [float(nm_confusion['TP'] + nm_confusion['TN']) / num_of_data,
                   nm_confusion['TP'] / (nm_confusion['TP'] + 0.5 * nm_confusion['FP'] + 0.5 * nm_confusion['FN'])]}


def check_int_range(value: str) -> int:
    """
    Check whether value is 0 or 1
    :param value: string value
    :return: integer value
    """
    int_value = int(value)
    if int_value != 0 and int_value != 1:
        raise argparse.ArgumentTypeError(f'"{value}" is an invalid value. It should be 0 or 1.')
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


def parse_arguments():
    """
    Parse all arguments
    :return: arguments
    """
    parser = argparse.ArgumentParser(description='Logistic regression')
    parser.add_argument('-m', '--mode', help='0: cross validation, 1: prediction', default=0, type=check_int_range)
    parser.add_argument('-v', '--verbosity', help='verbosity level (0-1)', default=0, type=check_int_range)

    return parser.parse_args()


if __name__ == '__main__':
    """
    Main function
        command: python3 logistic_regression.py [-m (0-1)] [-v (0-1)]
    """
    pd.set_option('display.max_rows', None)
    pp = pprint.PrettyPrinter()

    # Parse arguments
    args = parse_arguments()
    mode = args.mode
    verbosity = args.verbosity

    # Get training Info sheet
    info_log('=== Loading training data ===')
    tr_info = pd.read_excel('data/Training data.xlsx', sheet_name='Info',
                            names=['No', 'Gender', 'Age', 'Comorbidities', 'Antibiotics', 'Bacteria', 'Target'])
    # Get training TPR sheet
    tr_tpr = pd.read_excel('data/Training data.xlsx', sheet_name='TPR')

    if not mode:
        info_log('=== Cross validation ===')

        # Preprocess Info sheet
        tr_info = cv_info_fixer(tr_info)

        # Preprocess TPR sheet
        tr_tpr = cv_tpr_fixer(tr_tpr)

        # Merge Info and TPR
        tr_data = pd.merge(tr_info, tr_tpr, on='No')

        # Get training target
        tr_target = tr_data['Target'].copy()
        del tr_data['Target']
        del tr_data['No']

        cross_validator(tr_data, tr_target)
    else:
        info_log('=== Prediction ===')

        # Get submission
        sub = pd.read_csv('data/Submission.csv')

        # Get testing Info sheet
        info_log('=== Loading testing data ===')
        ts_info = pd.read_excel('data/testing_data.xlsx', sheet_name='Info',
                                names=['No', 'Gender', 'Age', 'Comorbidities', 'Antibiotics', 'Bacteria'])
        # Merge ts_info and sub
        ts_info = pd.merge(ts_info, sub, on='No')
        # Get testing TPR sheet
        ts_tpr = pd.read_excel('data/testing_data.xlsx', sheet_name='TPR')

        # Preprocess training and testing Info
        tr_info, ts_info = predict_info_fixer(tr_info, ts_info)

        # Preprocess training and testing TPR
        tr_tpr, ts_tpr = predict_tpr_fixer(tr_tpr, ts_tpr)

        # Merge Info and TPR
        tr_data, ts_data = pd.merge(tr_info, tr_tpr, on='No'), pd.merge(ts_info, ts_tpr, on='No')

        # Get training target
        tr_target = tr_data['Target'].copy()
        del tr_data['Target']
        del tr_data['No']
        del ts_data['Target']
