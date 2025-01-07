import pandas as pd
import streamlit as st
from normalization import drop_columns, fill_mice, fill_knn, fill_mean, normalize, encode, remove_outliers, \
    remove_outliers_zscore

# Load the dataset
file_path = "global_health.csv"
df = pd.read_csv(file_path)

# df = df[~df["Country"].isin(["India", "China", "Monaco"])]

# Default features
default_features = ["Water_Access_Percent", "Country_Code", "Country", "Life_Expectancy_Female",
                    "Life_Expectancy_Male", "Female_Population", "Male_Population"]

fill_value_option = ["Mean", "KNN", "MICE"]

# Initialize session state for multiselect and radio buttons
if "selected_features" not in st.session_state:
    st.session_state.selected_features = default_features

if "selected_algorithm" not in st.session_state:
    st.session_state.selected_algorithm = fill_value_option[0]  # Default to "Mean"

# Multiselect widget with session state
selected_features = st.multiselect(
    'Select countries for comparison',
    options=df.columns,
    default=st.session_state.selected_features
)

# Update session state when multiselect value changes
if selected_features != st.session_state.selected_features:
    st.session_state.selected_features = selected_features

# Drop selected columns
to_drop = st.session_state.selected_features
df = drop_columns(df, to_drop)

# Title
st.title("Normalized and Cleaned Data")

# Radio button widget with session state
selected_algorithm = st.radio(
    "Select method to fill missing values:",
    fill_value_option,
    index=fill_value_option.index(st.session_state.selected_algorithm),
    horizontal=True
)

# Update session state when radio value changes
if selected_algorithm != st.session_state.selected_algorithm:
    st.session_state.selected_algorithm = selected_algorithm

# Apply the selected filling algorithm
if st.session_state.selected_algorithm == "Mean":
    df = fill_mean(df)
elif st.session_state.selected_algorithm == "KNN":
    df = fill_knn(df)
else:
    df = fill_mice(df)

# Display cleaned data
st.write("Clean data with missing data filled")
st.write(df)

# Save cleaned data to a file
df.to_csv("clean_data.csv", index=False)

# Normalize the data
df = normalize(df)

# Display normalized data
st.write("Normalized data")
st.write(df)

# Save normalized data to a file
df.to_csv("normalized.csv", index=False)
