import pandas as pd
import matplotlib.pyplot as plt
import pprint
import sys
import numpy as np
from math import floor, ceil
from argparse import ArgumentParser, ArgumentTypeError, Namespace
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.model_selection import RepeatedStratifiedKFold
from typing import List, Tuple, Dict
from statistics import mean
from sklearn.metrics import f1_score, accuracy_score
from logistic_regression import LogisticRegression


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
    Use cross validation to test logistic regression
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
    methods = ['Gradient descent']
    accuracy = {name: {f'{i}': [] for i in range(2, total_features)} for name in methods}
    f1 = {name: {f'{i}': [] for i in range(2, total_features)} for name in methods}
    weights = {name: {f'{i}': [] for i in range(2, total_features)} for name in methods if name != 'sklearn'}
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
            lr = LogisticRegression(learning_rate=learning_rate, regularization=regularization, penalty=penalty)
            weight = lr.fit(concatenated_train)
            weights['Gradient descent'][f'{k}'].append(weight)

            # Classify testing data
            prediction = lr.predict(weight, data_test)
            acc, score = accuracy_score(target_test, prediction), f1_score(target_test, prediction)
            accuracy['Gradient descent'][f'{k}'].append(acc)
            f1['Gradient descent'][f'{k}'].append(score)

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

    # Plot weights
    plot_weights(weights)

    plt.tight_layout()
    plt.show()


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


def plot_weights(weights: Dict[str, Dict[str, List[np.ndarray]]]) -> None:
    """
    Plot feature weights
    :param weights: weights of Gradient descent and Newton's method with different features
    :return: None
    """
    fig = plt.figure(2)
    fig.canvas.set_window_title('Average Feature Weight')

    rows, cols = len(weights.keys()), len(list(weights.values())[0].keys())

    count = 1
    for name, features in weights.items():
        for num_of_features, weight in features.items():
            ax = fig.add_subplot(rows, cols, count)

            mean_weight = np.mean(np.array(weight), axis=0)
            mean_weight = mean_weight.reshape(len(mean_weight))

            colors = ['g' if value > 0 else 'r' for value in mean_weight[1:]]

            y_pos = np.arange(int(num_of_features))

            ax.barh(y_pos, mean_weight[1:], align='center', color=colors)
            ax.set_xlim(floor(min(mean_weight[1:])), ceil(max(mean_weight[1:])))
            ax.set_yticks(y_pos)
            ax.set_yticklabels(y_pos)
            ax.invert_yaxis()
            ax.set_title(f'{num_of_features} features')

            count += 1


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
        testing_info = pd.read_excel('data/Testing data.xlsx', sheet_name='Info',
                                     names=['No', 'Gender', 'Age', 'Comorbidities', 'Antibiotics', 'Bacteria'])
        # Merge ts_info and sub
        testing_info = pd.merge(testing_info, submission, on='No')
        # Get testing TPR sheet
        testing_tpr = pd.read_excel('data/Testing data.xlsx', sheet_name='TPR')

        # Preprocess training and testing Info
        training_info, testing_info = predict_info_fixer(training_info, testing_info)

        # Preprocess training and testing TPR
        training_tpr, testing_tpr = predict_tpr_fixer(training_tpr, testing_tpr)

        # Merge Info and TPR
        training_data = pd.merge(training_info, training_tpr, on='No')
        testing_data = pd.merge(testing_info, testing_tpr, on='No')
        del training_data['No']

        # Separate training data and training target
        train_data = training_data.copy()
        train_target = training_data['Target']
        del train_data['Target']

        # Get features
        features = feature_selection(train_data, train_target, 5)

        # Train the model
        lr = LogisticRegression(learning_rate=learning_rate, regularization=regularization, penalty=penalty)
        weight = lr.fit(training_data[features + ['Target']])

        # Predict the results
        prediction = lr.predict(weight=weight, test_data=testing_data)
        pred_prob = lr.pred_probability(weight=weight, test_data=testing_data)

        # Construct results
        result = pd.DataFrame(list(zip(testing_data['No'], list(prediction))), columns=['No', 'Target'])
        prob_result = pd.DataFrame(list(zip(testing_data['No'], list(prediction))), columns=['No', 'Target'])
        info_log('Results')
        if verbosity:
            print(result)
        info_log('Probability results')
        if verbosity:
            print(prob_result)

        # Write results to csv
        result.to_csv('result.csv', index=False)


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
