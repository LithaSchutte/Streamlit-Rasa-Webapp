import streamlit as st
import pandas as pd
import time
import os
from generate_fake_data import generate_fake_data, add_fake_data_to_real_data


def load_data(file_path):
    try:
        data = pd.read_csv(file_path)
        return data
    except FileNotFoundError:
        st.error(f"File not found: {file_path}")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

st.set_page_config(layout="wide")

# Title and Description
st.title("Synthetic Data Generation")
st.write(
    "Synthetic data is automatically generated when the application is started for necessary functionalities in other "
    "parts of the application. However, as a user, you are welcome to regenerate synthetic data as often as you wish. "
    "Synthetic data will always be randomized. Visit the data page to view the original dataset with the added synthetic data"
)

st.write("Sample of the synthetic data:")

progress_placeholder = st.empty()

real_data = load_data('global_health.csv')

fake_data_file = 'fake_data.csv'

data_placeholder = st.empty()

fake_data = None

if os.path.exists(fake_data_file):
    fake_data = load_data(fake_data_file)
    data_placeholder.dataframe(fake_data.head())


col1, col2, col3 = st.columns([1, 1, 1.5], gap="small")

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
        fake_data = load_data(fake_data_file)
        st.write(fake_data)  # Show the full dataset if requested
