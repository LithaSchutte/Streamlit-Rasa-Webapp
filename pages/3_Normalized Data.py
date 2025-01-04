import pandas as pd
import streamlit as st
from normalization import drop_columns, fill_mice, fill_knn, fill_mean, normalize

file_path = "global_health.csv"

df = pd.read_csv(file_path)

to_drop = ["Water_Access_Percent", "Hospital_Beds_Per_1000", "Suicide_Rate_Percent", "Country_Code", "Country",
           "Labour_Force_Total", "CO2_Exposure_Percent", "Unemployment_Rate", "Life_Expectancy_Female",
           "Life_Expectancy_Male", "Female_Population", "Male_Population"]

fill_value_option = ["Mean", "KNN", "MICE"]

df = drop_columns(df, to_drop)

st.title("Normalized and Cleaned Data")

selected_algorithm = st.radio("Select method to fill missing values:", fill_value_option, horizontal=True)

is_mean = selected_algorithm == "Mean"
is_knn = selected_algorithm == "KNN"
is_mice = selected_algorithm == "MICE"

if is_mean:
    df = fill_mean(df)
elif is_knn:
    df = fill_knn(df)
else:
    df = fill_mice(df)

st.write("Clean data with missing data filled")
st.write(df)

df.to_csv("clean_data.csv", index=False)

df = normalize(df)

st.write("Normalized data")
st.write(df)

df.to_csv("normalized.csv", index=False)