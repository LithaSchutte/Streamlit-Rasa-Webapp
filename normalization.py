from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def drop_columns(dataset, columns):
    dataset = dataset.drop(columns=columns)
    return dataset

def encode(dataset, column):
    unique_values = sorted(dataset[column].unique())
    mapping = {value: idx for idx, value in enumerate(unique_values, start=1)}
    dataset[column] = dataset[column].map(mapping)
    return dataset

def fill_mean(dataset):
    dataset = dataset.fillna(dataset.mean())
    return dataset

def fill_knn(dataset):
    knn = KNNImputer(n_neighbors=5)
    imputed_data = pd.DataFrame(knn.fit_transform(dataset), columns=dataset.columns)
    return imputed_data

def fill_mice(dataset):
    mice = IterativeImputer(max_iter=10, random_state=0)
    imputed_data = pd.DataFrame(mice.fit_transform(dataset), columns=dataset.columns)
    return imputed_data

def remove_outliers(dataset, excluded):
    for column in dataset.select_dtypes(include=['number']).columns:
        if column in excluded:
            continue
        Q1 = dataset[column].quantile(0.25)
        Q3 = dataset[column].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        dataset = dataset[(dataset[column] >= lower_bound) & (dataset[column] <= upper_bound)]

    return dataset

def remove_outliers_zscore(dataset, threshold, excluded):
    for column in dataset.select_dtypes(include=['number']).columns:
        if column in excluded:
            continue
        z_scores = (dataset[column] - dataset[column].mean()) / dataset[column].std()
        dataset[column] = dataset[column].apply(lambda x: dataset[column].max() if abs((x - dataset[column].mean()) / dataset[column].std()) > threshold else x)
    return dataset

def normalize(dataset):
    scaler = StandardScaler()
    num_cols = dataset.select_dtypes(include=["float64", "int64"]).columns
    dataset[num_cols] = scaler.fit_transform(dataset[num_cols])
    return dataset
