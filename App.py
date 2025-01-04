import os
import pandas as pd
import streamlit as st
from generate_fake_data import generate_fake_data, add_fake_data_to_real_data
from normalization import drop_columns, fill_mice, normalize
import plotly.express as px


st.title("🌍 Global Health & Development")
data_file = 'global_health.csv'


@st.cache_data
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

original_data = load_data(data_file)

if 'fake_data.csv' not in os.listdir() and 'real_data_with_added_fake_data.csv' not in os.listdir():
    fake_data = generate_fake_data(original_data, 1000)
    added_fake_data = add_fake_data_to_real_data(original_data, fake_data)

    fake_data.to_csv('fake_data.csv', index=False)
    added_fake_data.to_csv('real_data_with_added_fake_data.csv', index=False)

if 'normalized.csv' not in os.listdir() and 'clean_data.csv' not in os.listdir():
    # Columns to drop
    to_drop = ["Water_Access_Percent", "Hospital_Beds_Per_1000", "Suicide_Rate_Percent", "Country_Code", "Country",
               "Year", "Labour_Force_Total", "CO2_Exposure_Percent", "Unemployment_Rate", "Life_Expectancy_Female",
               "Life_Expectancy_Male", "Female_Population", "Male_Population", "Total_Population"]

    # Drop unnecessary columns
    cleaned_data = drop_columns(original_data, to_drop)

    # Fill missing values using MICE
    filled_data = fill_mice(cleaned_data)

    # Normalize the data
    normalized_data = normalize(filled_data)

    # Save the processed files
    normalized_data.to_csv('normalized.csv', index=False)
    filled_data.to_csv('cleaned_data.csv', index=False)

st.write("This app focuses on global health development by leveraging regression models to predict life expectancy across "
         "different countries. By analyzing various socio-economic, environmental, and healthcare-related factors, the "
         "app provides insights into how these variables influence life expectancy. The app aims to support data-driven "
         "decision-making in global health initiatives, offering valuable predictions that can guide policymakers and "
         "researchers in improving public health outcomes worldwide.")

unique_countries = original_data['Country'].unique()

fig = px.choropleth(locations=unique_countries,
                    locationmode='country names',
                    color=unique_countries,  # Coloring based on country names
                    title='Data from the following countries are considered:')

fig.update_layout(coloraxis_showscale=False)

st.plotly_chart(fig)

