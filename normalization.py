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

def normalize(dataset):
    scaler = MinMaxScaler()
    num_cols = dataset.select_dtypes(include=["float64", "int64"]).columns
    dataset[num_cols] = scaler.fit_transform(dataset[num_cols])
    return dataset

def normalize_lasso(dataset):
    scaler = StandardScaler()
    num_cols = dataset.select_dtypes(include=["float64", "int64"]).columns
    dataset[num_cols] = scaler.fit_transform(dataset[num_cols])
    return dataset
