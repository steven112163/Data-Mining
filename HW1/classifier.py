import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from typing import List
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
    skf = StratifiedKFold()
    acc = []
    for train_index, test_index in skf.split(training_data, training_target):
        # Get training set and testing set
        data_train, target_train = training_data.iloc[train_index.tolist()], training_target.iloc[train_index.tolist()]
        data_test, target_test = training_data.iloc[test_index.tolist()], training_target.iloc[test_index.tolist()]

        # Use training set to select features
        features = feature_selection(data_train, target_train, 2)

        # Use model to train and predict
        acc.append(k_nearest_neighbors(data_train, target_train, data_test, target_test, features))
    print(acc)
    print(mean(acc))


def k_nearest_neighbors(data_train: pd.DataFrame, target_train: pd.Series, data_test: pd.DataFrame,
                        target_test: pd.Series, features: List[str]) -> float:
    """

    :param data_train: training data set
    :param target_train: training target
    :param data_test: testing data set
    :param target_test: testing target
    :param features: selected features
    :return: accuracy
    """
    # Train and predict
    neigh = KNeighborsClassifier()
    neigh.fit(data_train[features], target_train)
    prediction = neigh.predict(data_test[features]).tolist()

    # Get accuracy
    diff = 0
    for idx, target in enumerate(target_test):
        if prediction[idx] == target:
            diff += 1

    return float(diff) / len(prediction)


if __name__ == '__main__':
    pd.set_option('display.max_rows', None)
    pp = pprint.PrettyPrinter()

    # Get Info sheet
    tr_info = pd.read_excel('training_data.xlsx', sheet_name='Info',
                            names=['No', 'Gender', 'Age', 'Comorbidities', 'Antibiotics', 'Bacteria', 'Target'])
    tr_info = info_fixer(tr_info)
    # print(tr_info)

    # Get TPR sheet
    tr_tpr = pd.read_excel('training_data.xlsx', sheet_name='TPR')
    tr_tpr = tpr_fixer(tr_tpr)
    # print(tr_tpr)

    # Merge Info and TPR
    tr_data = pd.merge(tr_info, tr_tpr, on='No')
    # print(tr_data)

    # Get training target
    tr_target = tr_data['Target'].copy()
    del tr_data['Target']
    del tr_data['No']

    # print(feature_selection(tr_data, tr_target, 2))
    cross_validator(tr_data, tr_target)
