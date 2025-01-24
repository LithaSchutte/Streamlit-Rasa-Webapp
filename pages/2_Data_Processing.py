import streamlit as st
import time
import os
from generate_fake_data import generate_fake_data, add_fake_data_to_real_data
from data_processing import drop_columns, fill_mice, fill_knn, fill_mean, normalize, handle_outliers
from AppClass import DataLoader  # Import the DataLoader class

st.set_page_config(layout="wide")

# Title and Description
st.title("Data Processing")
st.write("""### Synthetic data Generation
The application also allows for the generation of **synthetic data**:
- **Why Generate Synthetic data?**
  - Synthetic data helps in testing and developing models without using sensitive real-world data.
  - It mirrors real-world patterns while ensuring privacy and ethical use.
- **How It's Done**:
  - Numerical columns are randomized within the observed range of the dataset.
  - Logical rules ensure realism, such as maintaining a consistent gap between male and female life expectancy.
  - Categorical columns, like `Country`, are randomized based on predefined lists to reflect diversity.

You can regenerate synthetic data or view the complete synthetic dataset directly in the app.""")

st.write("Sample of the synthetic data:")

progress_placeholder = st.empty()

data_loader = DataLoader('data/global_health.csv')

real_data = data_loader.load_data()

fake_data_file = 'data/fake_data.csv'

data_placeholder = st.empty()

fake_data = None

if os.path.exists(fake_data_file):
    fake_data_loader = DataLoader(fake_data_file, cache_data=False)
    fake_data = fake_data_loader.load_data()
    data_placeholder.dataframe(fake_data.head())

col1, col2, col3 = st.columns([1, 1, 1], gap="small")

with col1:
    regenerate_button = st.button("Regenerate Synthetic data", key="regenerate_fake_data_button")
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

    fake_data.to_csv("data/fake_data.csv", index=False)
    combined_data.to_csv('data/real_data_with_added_fake_data.csv', index=False)

    to_drop = ["Water_Access_Percent", "Hospital_Beds_Per_1000", "Suicide_Rate_Percent", "Country_Code", "Country",
               "Year", "Labour_Force_Total", "CO2_Exposure_Percent", "Unemployment_Rate", "Life_Expectancy_Female",
               "Life_Expectancy_Male", "Female_Population", "Male_Population", "Total_Population", "Infant_Deaths"]

    clean_fake_data = drop_columns(combined_data, to_drop)
    filled_clean_fake_data = fill_mice(clean_fake_data)
    clean_fake_data_outliers = handle_outliers(filled_clean_fake_data, 3)
    clean_fake_data_outliers.to_csv("data/clean_fake_data.csv", index=False)

if fake_data is not None:
    if full_button:
        fake_data_loader = DataLoader(fake_data_file)
        fake_data = fake_data_loader.load_data()
        st.write(fake_data)  # Show the full dataset if requested


st.write("## Cleaning and Preprocessing")

st.write("""
### Feature Engineering
To ensure optimal performance and a clean dataset, some columns are dropped based on their relevance and relationship with the target variable. Here's how this process is handled:

- **Dropped Columns**:
  - `Life_Expectancy_Female` and `Life_Expectancy_Male` were removed because they overlap conceptually with the target variable, `Life_Expectancy`.
  - Features with **low correlation** to the target variable, such as `Country`, `Country_Code`, `Hospital_Beds_Per_1000`, `Labour_Force_Total` and `Suicide_Rate`, were excluded to reduce noise.
  - Columns like `CO2_Exposure_Percent` were redundant, as they were identical to other features (e.g., `Air_Pollution`).
  - `Water_Access_Percent` had a high number of missing values and conceptually overlapped with `Safe_Water_Access_Percent`, which was retained for better representation.

This reduces unnecessary data, making the dataset more efficient for analysis and modeling.""")

st.write("""### Data Transformation
Once the dataset is refined, further transformations are applied:
- **Handling Missing Values**: As a user you can select from three methods to fill missing data:
  - `Mean`: Simple average replacement, effective when data is missing at random.
  - `KNN`: Estimates missing values based on similar rows in the dataset.
  - `MICE`: Uses relationships between variables to impute missing values. (provides the most accurate results).)""")

fill_value_option = ["MICE", "Mean", "KNN"]

columns_to_drop = ["CO2_Exposure_Percent","Total_Population", "Labour_Force_Total","Water_Access_Percent", "Country_Code", "Country", "Life_Expectancy_Female",
                   "Life_Expectancy_Male", "Female_Population", "Male_Population", "Suicide_Rate_Percent", "Hospital_Beds_Per_1000", "Infant_Deaths"]

if "selected_algorithm" not in st.session_state:
    st.session_state.selected_algorithm = fill_value_option[0]  # Default to "MICE"

selected_algorithm = st.radio(
    "Select method to fill missing values:",
    fill_value_option,
    index=fill_value_option.index(st.session_state.selected_algorithm),
    horizontal=True
)

if selected_algorithm != st.session_state.selected_algorithm:
    st.session_state.selected_algorithm = selected_algorithm
    st.rerun()

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

df.to_csv("data/clean_data.csv", index=False)
