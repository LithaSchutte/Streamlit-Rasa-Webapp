from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import pandas as pd
import streamlit as st
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler

file_path = "global_health.csv"

df = pd.read_csv(file_path)

to_drop = ["Water_Access_Percent", "Hospital_Beds_Per_1000", "Suicide_Rate_Percent", "Country_Code", "Country",
           "Year", "Labour_Force_Total", "CO2_Exposure_Percent", "Unemployment_Rate", "Life_Expectancy_Female",
           "Life_Expectancy_Male", "Female_Population", "Male_Population", "Total_Population"]

fill_value_option = ["Mean", "KNN", "MICE"]

def drop_columns(dataset, columns):
    dataset = dataset.drop(columns=columns)
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

st.write("### Original Dataset")
st.write(df)
df = drop_columns(df, to_drop)
st.write("### Dropped columns")
st.write(df)

selected_algorithm = st.radio("Select how to fill missing values:", fill_value_option, horizontal=True)

is_mean = selected_algorithm == "Mean"
is_knn = selected_algorithm == "KNN"
is_mice = selected_algorithm == "MICE"

if is_mean:
    df = fill_mean(df)
    st.write("Mean")
elif is_knn:
    df = fill_knn(df)
    st.write("KNN")
else:
    df = fill_mice(df)
    st.write("MICE")

st.write("Filled missing values")
st.write(df)

df = normalize(df)

st.write("Normalized")
st.write(df)

df.to_csv("normalized.csv", index=False)
