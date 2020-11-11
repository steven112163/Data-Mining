import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.metrics import f1_score, accuracy_score
from typing import List, Tuple
from statistics import mean
import pprint


def info_fixer(training_info: pd.DataFrame) -> pd.DataFrame:
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


def tpr_fixer(training_tpr: pd.DataFrame) -> pd.DataFrame:
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
    skf = StratifiedKFold(shuffle=True)
    accuracy = {'K Nearest Neighbors': {'1': [], '2': [], '3': []},
                'Naive Bayes': {'1': [], '2': [], '3': []},
                'Gaussian Process': {'1': [], '2': [], '3': []}}
    score = {'K Nearest Neighbors': {'1': [], '2': [], '3': []},
             'Naive Bayes': {'1': [], '2': [], '3': []},
             'Gaussian Process': {'1': [], '2': [], '3': []}}

    # Run 10 times to get the average
    for _ in range(10):
        for train_index, test_index in skf.split(training_data, training_target):
            # Get training set and testing set
            data_train, target_train = training_data.iloc[train_index.tolist()], training_target.iloc[
                train_index.tolist()]
            data_test, target_test = training_data.iloc[test_index.tolist()], training_target.iloc[test_index.tolist()]

            # Use training set to select features
            for k in range(1, 4):
                features = feature_selection(data_train, target_train, k)

                # Use knn to train and predict
                acc, f1 = k_nearest_neighbors(data_train, target_train, data_test, target_test, features)
                accuracy['K Nearest Neighbors'][f'{k}'].append(acc)
                score['K Nearest Neighbors'][f'{k}'].append(f1)

                # Use naive bayes to train and predict
                acc, f1 = naive_bayes(data_train, target_train, data_test, target_test, features)
                accuracy['Naive Bayes'][f'{k}'].append(acc)
                score['Naive Bayes'][f'{k}'].append(f1)

                # Use gaussian process to train and predict
                acc, f1 = gaussian_process(data_train, target_train, data_test, target_test, features)
                accuracy['Gaussian Process'][f'{k}'].append(acc)
                score['Gaussian Process'][f'{k}'].append(f1)

    # Print results
    print('=== Accuracy ===')
    for model, results in accuracy.items():
        print(f'=== {model} ===')
        for num_of_features, values in results.items():
            print(f'{num_of_features}: {mean(values)}')
        print()
    print('\n=== F1 score ===')
    for model, results in score.items():
        print(f'=== {model} ===')
        for num_of_features, values in results.items():
            print(f'{num_of_features}: {mean(values)}')
        print()


def k_nearest_neighbors(data_train: pd.DataFrame, target_train: pd.Series, data_test: pd.DataFrame,
                        target_test: pd.Series, features: List[str]) -> Tuple[float, float]:
    """
    K Nearest Neighbors Classifier
    :param data_train: training data set
    :param target_train: training target
    :param data_test: testing data set
    :param target_test: testing target
    :param features: selected features
    :return: accuracy and f1 score
    """
    # Train and predict
    neigh = KNeighborsClassifier().fit(data_train[features], target_train)
    prediction = neigh.predict(data_test[features]).tolist()

    # Return accuracy and f1 score
    return accuracy_score(prediction, target_test), f1_score(prediction, target_test)


def naive_bayes(data_train: pd.DataFrame, target_train: pd.Series, data_test: pd.DataFrame,
                target_test: pd.Series, features: List[str]) -> Tuple[float, float]:
    """
    Naive Bayes Classifier
    :param data_train: training data set
    :param target_train: training target
    :param data_test: testing data set
    :param target_test: testing target
    :param features: selected features
    :return: accuracy and f1 score
    """
    # Train and predict
    nb = GaussianNB().fit(data_train[features], target_train)
    prediction = nb.predict(data_test[features])

    # Return accuracy and f1 score
    return accuracy_score(prediction, target_test), f1_score(prediction, target_test)


def gaussian_process(data_train: pd.DataFrame, target_train: pd.Series, data_test: pd.DataFrame,
                     target_test: pd.Series, features: List[str]) -> Tuple[float, float]:
    """
    Gaussian Process Classifier
    :param data_train: training data set
    :param target_train: training target
    :param data_test: testing data set
    :param target_test: testing target
    :param features: selected features
    :return: accuracy and f1 score
    """
    # Train and predict
    nb = GaussianProcessClassifier().fit(data_train[features], target_train)
    prediction = nb.predict(data_test[features])

    # Return accuracy and f1 score
    return accuracy_score(prediction, target_test), f1_score(prediction, target_test)


if __name__ == '__main__':
    pd.set_option('display.max_rows', None)
    pp = pprint.PrettyPrinter()

    # Get Info sheet
    tr_info = pd.read_excel('training_data.xlsx', sheet_name='Info',
                            names=['No', 'Gender', 'Age', 'Comorbidities', 'Antibiotics', 'Bacteria', 'Target'])
    tr_info = info_fixer(tr_info)

    # Get TPR sheet
    tr_tpr = pd.read_excel('training_data.xlsx', sheet_name='TPR')
    tr_tpr = tpr_fixer(tr_tpr)

    # Merge Info and TPR
    tr_data = pd.merge(tr_info, tr_tpr, on='No')

    # Get training target
    tr_target = tr_data['Target'].copy()
    del tr_data['Target']
    del tr_data['No']

    cross_validator(tr_data, tr_target)
