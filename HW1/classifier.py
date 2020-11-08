import pandas as pd
from sklearn import preprocessing as pre


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
    new_info['Antibiotics'] = new_info.groupby('Antibiotics').ngroup()
    new_info['Bacteria'] = new_info.groupby('Bacteria').ngroup()

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
    patients[['T', 'P', 'R', 'NBPS', 'NBPD']] = pre.StandardScaler().fit_transform(
        patients[['T', 'P', 'R', 'NBPS', 'NBPD']])

    return patients


if __name__ == '__main__':
    pd.set_option('display.max_rows', None)

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
    print(tr_data)
