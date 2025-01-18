import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from AppClass import DataLoader

st.set_page_config(layout="wide")

st.title("Data Viewer and Insights")

options = ["Original Data", "Processed Data", "Data with Added Synthetic Data"]

selected_option = st.radio(
    "Select a dataset to display:",
    options,
    horizontal=True)

file_mapping = {
    "Original Data": "data/global_health.csv",
    "Processed Data": "data/clean_data.csv",
    "Data with Added Synthetic Data": "data/real_data_with_added_fake_data.csv"
}

# data file to load based on user selection
data_file = file_mapping.get(selected_option)

# Initialize data as empty
data_placeholder = st.empty()

# Reload data on each selection
if selected_option == "Data with Added Synthetic Data":
    fake_data_loader = DataLoader(data_file, cache_data=False)
    data = fake_data_loader.load_data()
elif selected_option == "Processed Data":
    processed_data_loader = DataLoader(data_file, cache_data=False)
    data = processed_data_loader.load_data()
else:
    real_data_loader = DataLoader(data_file, cache_data=True)
    data = real_data_loader.load_data()

data_placeholder = st.empty()

# Check if data is loaded successfully
if not data.empty:
    data_placeholder.dataframe(data)
    st.markdown("---")
    st.subheader("📈 Global Health Data Statistics")
    st.write("### Data Statistics")
    st.write(data.describe())
    st.write("### Additional Insights")
    st.write(f"Total entries: {len(data)}")
    st.write(f"Columns: {', '.join(data.columns)}")
    st.markdown("---")
    st.write("### Data Correlation")

    if selected_option == "Processed Data":
        # Normalized data View Options
        view_options = ["Table View", "Graph View"]
        selected_view = st.radio(
            "Select a view:",
            view_options,
            horizontal=True,
        )

        if selected_view == "Table View":
            st.write("Correlation Matrix")
            if 'ID' in data.columns:
                data_without_id = data.drop(columns=['ID'])
            else:
                data_without_id = data
            correlation_matrix = data_without_id.corr()
            st.write(correlation_matrix)
        elif selected_view == "Graph View":
            st.write("Graphical Representation")
            if 'ID' in data.columns:
                data_without_id = data.drop(columns=['ID'])
            else:
                data_without_id = data

            correlation_matrix = data_without_id.corr()

            fig, ax = plt.subplots(figsize=(8, 6))

            sns.heatmap(
                correlation_matrix,
                annot=True,
                cmap="coolwarm",
                fmt=".2f",
                linewidths=0.5,
                ax=ax
            )
            st.pyplot(fig)
    else:
        st.warning("Correlation analysis is only available for Processed Data.")


    st.write("## Data Visualization")
    if selected_option == "Original Data" or selected_option == "Data with Added Synthetic Data":

        default_countries = ['Germany', 'Italy', 'United States', 'Canada']

        selected_countries = st.multiselect(
            'Select countries for comparison',
            options=data['Country'].unique(),
            default=default_countries
        )

        if selected_countries:
            filtered_df = data[data['Country'].isin(selected_countries)]

            if selected_option == "Original Data": # don't showcase year graph if not original data, i.e. fake data will add duplicate years
                st.write("### Life Expectancy Over Years")
                plt.figure(figsize=(10, 6))
                for country in selected_countries:
                    country_data = filtered_df[filtered_df['Country'] == country]
                    plt.plot(country_data['Year'], country_data['Life_Expectancy'], label=country)

                plt.xlabel('Year')
                plt.ylabel('Life Expectancy')
                plt.title('Life Expectancy Comparison Over Years')
                plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
                st.pyplot(plt)

            # **Second Plot: Scatter plot with selectable x-axis**
            st.write("### Life Expectancy vs. Selected Variable")
            # Selectable x-axis options
            available_columns = [col for col in data.columns if col not in ['Country', 'Year', 'Life_Expectancy']]
            selected_x_column = st.selectbox(
                'Select the X-axis variable for scatter plot',
                options=available_columns,
                index=available_columns.index('Fertility_Rate') if 'Fertility_Rate' in available_columns else 0
            )

            plt.figure(figsize=(10, 6))
            for country in selected_countries:
                country_data = filtered_df[filtered_df['Country'] == country]
                plt.scatter(
                    country_data[selected_x_column],  # X-axis: User-selected column
                    country_data['Life_Expectancy'],  # Y-axis: Life Expectancy
                    label=country
                )

            plt.xlabel(selected_x_column.replace('_', ' '))  # Replace underscores for better readability
            plt.ylabel('Life Expectancy')
            plt.title(f'Life Expectancy vs. {selected_x_column.replace("_", " ")}')
            plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
            st.pyplot(plt)

            # **Third Plot: Bar chart for male and female life expectancy**
            st.write("### Male and Female Life Expectancy Comparison")
            avg_life_expectancy = filtered_df.groupby('Country')[['Life_Expectancy_Male', 'Life_Expectancy_Female']].mean()

            # Plot bar chart
            plt.figure(figsize=(10, 6))
            bar_width = 0.4
            x = range(len(avg_life_expectancy))

            plt.bar(x, avg_life_expectancy['Life_Expectancy_Male'], width=bar_width, label='Male Life Expectancy',
                    color='blue')
            plt.bar([p + bar_width for p in x], avg_life_expectancy['Life_Expectancy_Female'], width=bar_width,
                    label='Female Life Expectancy', color='pink')

            plt.xlabel('Country')
            plt.ylabel('Life Expectancy')
            plt.title('Comparison of Male and Female Life Expectancy')
            plt.xticks([p + bar_width / 2 for p in x], avg_life_expectancy.index, rotation=45)
            plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

            st.pyplot(plt)

        else:
            st.write("Please select at least one country for comparison.")
    else:
        st.warning("Data Visualisation is not possible for Processed Data because features are altered or dropped")

else:
    st.error("The dataset is empty or could not be loaded.")
