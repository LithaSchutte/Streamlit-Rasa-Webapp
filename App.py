import os
import streamlit as st
from generate_fake_data import generate_fake_data, add_fake_data_to_real_data
from normalization import drop_columns, fill_mice, handle_outliers
import plotly.express as px
from AppClass import DataLoader
import time

st.set_page_config(layout="centered")

st.title("🌍 Global Health & Development")
data_file = 'data/global_health.csv'
data_loader = DataLoader(file_path=data_file, cache_data=True)

original_data = data_loader.load_data()

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

st.write("## Main App Features")
st.write("""
- Get insights into data statistics
- Explore data visualisations with charts and graphs
- Compare between countries
- Learn how the data was preprocessed
- Discover of the effect of synthetic data
- Predict life expectancy with regression
- Evaluate and compare regression models
- Talk to the Rasa chatbot about the data and the app
""")

st.write("Hint: The Data Page is a good place to start to first learn more about the data 😄")

# create initial versions of fake data and clean data upon start

if not os.path.isfile('data/fake_data.csv') and not os.path.isfile('data/real_data_with_added_fake_data.csv'):
    # Show message and progress bar for data generation
    st.write("Generating fake data and cleaning data. This might take a few moments, please wait...")
    progress_bar = st.progress(0)

    fake_data = generate_fake_data(original_data, 1000)
    progress_bar.progress(25)  # Update progress bar
    time.sleep(1)  # Simulate some processing time

    added_fake_data = add_fake_data_to_real_data(original_data, fake_data)
    progress_bar.progress(50)  # Update progress bar
    time.sleep(1)  # Simulate some processing time

    fake_data.to_csv('data/fake_data.csv', index=False)
    added_fake_data.to_csv('data/real_data_with_added_fake_data.csv', index=False)
    progress_bar.progress(75)  # Update progress bar
    time.sleep(1)  # Simulate some processing time

    to_drop = ["Water_Access_Percent", "Hospital_Beds_Per_1000", "Suicide_Rate_Percent", "Country_Code", "Country",
               "Year", "Labour_Force_Total", "CO2_Exposure_Percent", "Unemployment_Rate", "Life_Expectancy_Female",
               "Life_Expectancy_Male", "Female_Population", "Male_Population", "Total_Population", "Infant_Deaths"]

    clean_fake_data = drop_columns(added_fake_data, to_drop)
    filled_clean_fake_data = fill_mice(clean_fake_data)
    clean_fake_data_outliers = handle_outliers(filled_clean_fake_data, 3)
    clean_fake_data_outliers.to_csv("data/clean_fake_data.csv", index=False)

if not os.path.isfile('data/clean_data.csv'):
    # Columns to drop
    to_drop = ["Water_Access_Percent", "Hospital_Beds_Per_1000", "Suicide_Rate_Percent", "Country_Code", "Country",
               "Year", "Labour_Force_Total", "CO2_Exposure_Percent", "Unemployment_Rate", "Life_Expectancy_Female",
               "Life_Expectancy_Male", "Female_Population", "Male_Population", "Total_Population", "Infant_Deaths"]

    cleaned_data = drop_columns(original_data, to_drop)
    progress_bar.progress(85)  # Update progress bar
    time.sleep(1)  # Simulate some processing time

    filled_data = fill_mice(cleaned_data)
    progress_bar.progress(90)  # Update progress bar
    time.sleep(1)  # Simulate some processing time

    filled_data_outliers = handle_outliers(filled_data, 3)
    filled_data_outliers.to_csv('data/clean_data.csv', index=False)
    progress_bar.progress(100)  # Final progress bar update
    time.sleep(1)  # Simulate final processing time
    st.write("Data generation and cleaning complete!")
