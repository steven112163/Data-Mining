import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from typing import List, Tuple, Dict
from statistics import mean
import matplotlib.pyplot as plt
import pprint
import argparse
import numpy as np


def tr_info_fixer(training_info: pd.DataFrame) -> pd.DataFrame:
    """
    Fix problems in Info training set, e.g. missing values and categorical values etc.
    :param training_info: Info training set
    :return: new pandas DataFrame
    """
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


def tr_tpr_fixer(training_tpr: pd.DataFrame) -> pd.DataFrame:
    """
    Fix problems in TPR training set, e.g. missing values etc.
    :param training_tpr: TPR training set
    :return: new pandas DataFrame
    """
    # Group all data by patient number
    sectors = training_tpr.groupby('No')

    # Get means of each patient
    patients = sectors.mean()

    # Standardize all columns
    patients[['T', 'P', 'R', 'NBPS', 'NBPD']] = StandardScaler().fit_transform(
        patients[['T', 'P', 'R', 'NBPS', 'NBPD']])

    return patients


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
    # Setup K fold
    skf = RepeatedStratifiedKFold(n_repeats=10, random_state=0)

    # Setup accuracy and f1-score dictionary
    total_features = len(list(training_data)) + 1
    classifiers = ['Naive Bayes', 'Support Vector Machine', 'Decision Tree', 'Vote']
    accuracy = {name: {f'{i}': [] for i in range(2, total_features)} for name in classifiers}
    score = {name: {f'{i}': [] for i in range(2, total_features)} for name in classifiers}
    confusion = {name: {f'{i}': [] for i in range(2, total_features)} for name in classifiers}

    # Run 10 times to get the average
    for train_index, test_index in skf.split(training_data, training_target):
        # Get training set and testing set
        data_train, target_train = training_data.iloc[train_index.tolist()], training_target.iloc[
            train_index.tolist()]
        data_test, target_test = training_data.iloc[test_index.tolist()], training_target.iloc[test_index.tolist()]

        # Use training set to select features
        for k in range(2, total_features):
            features = feature_selection(data_train, target_train, k)

            # Use naive bayes to train and predict
            acc, f1, nb_predict, conf = naive_bayes(data_train, target_train, data_test, target_test, features)
            accuracy['Naive Bayes'][f'{k}'].append(acc)
            score['Naive Bayes'][f'{k}'].append(f1)
            confusion['Naive Bayes'][f'{k}'].append(conf)

            # Use support vector machine to train and predict
            acc, f1, svm_predict, conf = support_vector_machine(data_train, target_train, data_test, target_test,
                                                                features)
            accuracy['Support Vector Machine'][f'{k}'].append(acc)
            score['Support Vector Machine'][f'{k}'].append(f1)
            confusion['Support Vector Machine'][f'{k}'].append(conf)

            # Use decision tree to train and predict
            acc, f1, dt_predict, conf = decision_tree(data_train, target_train, data_test, target_test, features)
            accuracy['Decision Tree'][f'{k}'].append(acc)
            score['Decision Tree'][f'{k}'].append(f1)
            confusion['Decision Tree'][f'{k}'].append(conf)

            # Get results voted by three models
            results = list(zip(nb_predict, svm_predict, dt_predict))
            voted = []
            for i in results:
                voted.append(max(set(i), key=i.count))
            accuracy['Vote'][f'{k}'].append(accuracy_score(voted, target_test))
            score['Vote'][f'{k}'].append(f1_score(voted, target_test))

    # Print results and plot
    print('=== Accuracy ===')
    plt.subplot(121)
    plt.title('Accuracy')
    plot_and_print(accuracy)

    print('\n=== F1 score ===')
    plt.subplot(122)
    plt.title('F1 score')
    plot_and_print(score)

    print('\n=== Confusion matrix ===')
    for model, results in confusion.items():
        if model == 'Vote':
            continue
        print(f'=== {model} ===')
        for num_of_features, values in results.items():
            mean_confusion = sum(values) / 10.0
            print(f'{num_of_features}:')
            pp.pprint(mean_confusion)
        print()

    plt.tight_layout()
    plt.show()


def naive_bayes(data_train: pd.DataFrame, target_train: pd.Series, data_test: pd.DataFrame,
                target_test: pd.Series, features: List[str]) -> Tuple[float, float, List[int], np.ndarray]:
    """
    Naive Bayes Classifier
    :param data_train: training data set
    :param target_train: training target
    :param data_test: testing data set
    :param target_test: testing target
    :param features: selected features
    :return: accuracy, f1 score, prediction and confusion matrix
    """
    # Train and predict
    nb = GaussianNB().fit(data_train[features], target_train)
    prediction = nb.predict(data_test[features])

    # Return accuracy, f1 score and prediction
    return accuracy_score(prediction, target_test), f1_score(prediction, target_test), list(
        prediction), confusion_matrix(target_test, prediction)


def support_vector_machine(data_train: pd.DataFrame, target_train: pd.Series, data_test: pd.DataFrame,
                           target_test: pd.Series, features: List[str]) -> Tuple[float, float, List[int], np.ndarray]:
    """
    Support Vector Machine (classification)
    :param data_train: training data set
    :param target_train: training target
    :param data_test: testing data set
    :param target_test: testing target
    :param features: selected features
    :return: accuracy, f1 score, prediction and confusion matrix
    """
    # Train and predict
    svm = SVC(kernel='linear').fit(data_train[features], target_train)
    prediction = svm.predict(data_test[features])

    # Return accuracy, f1 score and prediction
    return accuracy_score(prediction, target_test), f1_score(prediction, target_test), list(
        prediction), confusion_matrix(target_test, prediction)


def decision_tree(data_train: pd.DataFrame, target_train: pd.Series, data_test: pd.DataFrame,
                  target_test: pd.Series, features: List[str]) -> Tuple[float, float, List[int], np.ndarray]:
    """
    Decision Tree Classifier
    :param data_train: training data set
    :param target_train: training target
    :param data_test: testing data set
    :param target_test: testing target
    :param features: selected features
    :return: accuracy, f1 score, prediction and confusion matrix
    """
    # Train and predict
    dt = DecisionTreeClassifier(max_depth=2).fit(data_train[features], target_train)
    prediction = dt.predict(data_test[features])

    # Return accuracy, f1 score and prediction
    return accuracy_score(prediction, target_test), f1_score(prediction, target_test), list(
        prediction), confusion_matrix(target_test, prediction)


def plot_and_print(score: Dict[str, Dict[str, List[float]]]) -> None:
    """
    Plot and print the results in score
    :param score: Dictionary of every classifier's results
    :return: None
    """
    for model, results in score.items():
        print(f'=== {model} ===')
        mean_values = []
        for num_of_features, values in results.items():
            mean_values.append(mean(values))
            print(f'{num_of_features}: {mean(values)}')
        print()
        plt.plot(list(results.keys()), mean_values, label=f'{model}')
    plt.ylim(0.0, 1.0)
    plt.legend()


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


def parse_arguments():
    """
    Parse all arguments
    :return: arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', help='0: cross validation, 1: prediction', default=0, type=check_int_range)

    return parser.parse_args()


if __name__ == '__main__':
    """
    Main function
    command: python3 classifier.py [-m (0-1)]
    """
    pd.set_option('display.max_rows', None)
    pp = pprint.PrettyPrinter()

    args = parse_arguments()
    mode = args.mode

    if not mode:
        # Get Info sheet
        tr_info = pd.read_excel('training_data.xlsx', sheet_name='Info',
                                names=['No', 'Gender', 'Age', 'Comorbidities', 'Antibiotics', 'Bacteria', 'Target'])
        tr_info = tr_info_fixer(tr_info)

        # Get TPR sheet
        tr_tpr = pd.read_excel('training_data.xlsx', sheet_name='TPR')
        tr_tpr = tr_tpr_fixer(tr_tpr)

        # Merge Info and TPR
        tr_data = pd.merge(tr_info, tr_tpr, on='No')

        # Get training target
        tr_target = tr_data['Target'].copy()
        del tr_data['Target']
        del tr_data['No']

        cross_validator(tr_data, tr_target)
