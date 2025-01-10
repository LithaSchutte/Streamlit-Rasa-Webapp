import streamlit as st
import time
import os
from generate_fake_data import generate_fake_data, add_fake_data_to_real_data
from normalization import drop_columns, fill_mice, fill_knn, fill_mean, normalize
from AppClass import DataLoader  # Import the DataLoader class

st.set_page_config(layout="wide")

# Title and Description
st.title("Data Processing")
st.write("## Synthetic Data Generation")
st.write(
    "Synthetic data is automatically generated when the application is started for necessary functionalities in other "
    "parts of the application. However, as a user, you are welcome to regenerate synthetic data as often as you wish. "
    "Synthetic data will always be randomized. Visit the data page to view the original dataset with the added synthetic data"
)

st.write("Sample of the synthetic data:")

progress_placeholder = st.empty()

# Initialize DataLoader for real data
data_loader = DataLoader('global_health.csv')

# Load the real data using DataLoader
real_data = data_loader.load_data()

fake_data_file = 'fake_data.csv'

# Placeholder for displaying fake data
data_placeholder = st.empty()

fake_data = None

if os.path.exists(fake_data_file):
    # Load fake data if the file exists
    fake_data_loader = DataLoader(fake_data_file)
    fake_data = fake_data_loader.load_data()
    data_placeholder.dataframe(fake_data.head())

col1, col2, col3 = st.columns([1, 1, 1], gap="small")

with col1:
    regenerate_button = st.button("Regenerate Synthetic Data", key="regenerate_fake_data_button")
with col2:
    full_button = st.button("Show Complete Synthetic Dataset", key="show_complete_data_button")

if regenerate_button:
    progress_text = "Regenerating synthetic data... Please wait."
    my_bar = progress_placeholder.progress(0, text=progress_text)

    for percent_complete in range(0, 100, 10):
        time.sleep(0.1)
        my_bar.progress(percent_complete + 10, text=progress_text)

    fake_data = generate_fake_data(real_data, num_new_rows=1000)
    combined_data = add_fake_data_to_real_data(real_data, fake_data)

    my_bar.progress(100, text="Synthetic data generation completed.")
    time.sleep(0.5)
    progress_placeholder.empty()
    st.success("Synthetic data has been generated successfully!")

    data_placeholder.dataframe(fake_data.head())

    fake_data.to_csv(fake_data_file, index=False)
    combined_data.to_csv('real_data_with_added_fake_data.csv', index=False)

if fake_data is not None:
    if full_button:
        fake_data_loader = DataLoader(fake_data_file)
        fake_data = fake_data_loader.load_data()
        st.write(fake_data)  # Show the full dataset if requested


st.write("## Cleaning and Preprocessing")
fill_value_option = ["MICE", "Mean", "KNN"]

columns_to_drop = ["CO2_Exposure_Percent","Total_Population", "Labour_Force_Total","Water_Access_Percent", "Country_Code", "Country", "Life_Expectancy_Female",
                   "Life_Expectancy_Male", "Female_Population", "Male_Population", "Suicide_Rate_Percent", "Hospital_Beds_Per_1000"]

if "selected_algorithm" not in st.session_state:
    st.session_state.selected_algorithm = fill_value_option[0]  # Default to "MICE"

selected_algorithm = st.radio(
    "Select method to fill missing values:",
    fill_value_option,
    index=fill_value_option.index(st.session_state.selected_algorithm),
    horizontal=True
)

# Update session state when radio value changes
if selected_algorithm != st.session_state.selected_algorithm:
    st.session_state.selected_algorithm = selected_algorithm

# Drop unnecessary columns
df = drop_columns(real_data, columns_to_drop)

# Apply the selected filling algorithm
if st.session_state.selected_algorithm == "Mean":
    df = fill_mean(df)
elif st.session_state.selected_algorithm == "KNN":
    df = fill_knn(df)
elif st.session_state.selected_algorithm == "MICE":
    df = fill_mice(df)

st.write("Clean data with missing data filled")
st.write(df)

df.to_csv("clean_data.csv", index=False)

# Normalize the data
df = normalize(df)
st.write("Normalized data")
st.write(df)
df.to_csv("normalized.csv", index=False)
