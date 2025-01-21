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
    column_bounds = {col: (dataset[col].min(skipna=True), dataset[col].max(skipna=True)) for col in dataset.columns}

    # Configure and apply the Iterative Imputer
    mice = IterativeImputer(max_iter=20, random_state=0)
    imputed_data = pd.DataFrame(mice.fit_transform(dataset), columns=dataset.columns)

    # Clip imputed values to the bounds of each column
    for col, (min_val, max_val) in column_bounds.items():
        imputed_data[col] = imputed_data[col].clip(lower=min_val, upper=max_val)

    return imputed_data

def handle_outliers(dataset, threshold):
    outliers = []  # To store all outliers with column names

    for column in dataset.columns:
        mean = dataset[column].mean()
        std_dev = dataset[column].std()

        def cap_outlier(value):
            z_score = (value - mean) / std_dev
            if z_score > threshold:
                outliers.append((column, value))
                return mean + threshold * std_dev
            elif z_score < -threshold:
                outliers.append((column, value))
                return mean - threshold * std_dev
            return value

        dataset[column] = dataset[column].apply(cap_outlier)

    return dataset




def normalize(dataset):
    scaler = StandardScaler()
    num_cols = dataset.select_dtypes(include=["float64", "int64"]).columns
    dataset[num_cols] = scaler.fit_transform(dataset[num_cols])
    return dataset
