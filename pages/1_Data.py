import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# Cache the data loading function
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

st.set_page_config(layout="wide")

st.title("Data Viewer and Insights")

options = ["Original Data", "Normalized Data", "Data with Fake Data"]

selected_option = st.radio(
    "Select a dataset to display:",
    options,
    horizontal=True,
)

st.write(selected_option)

file_mapping = {
    "Original Data": "global_health.csv",
    "Normalized Data": "clean_normalized_data.csv",
    "Data with Fake Data": "fake_data.csv"
}

data_file = file_mapping.get(selected_option)

data = load_data(data_file)

if not data.empty:
    st.write(data)

    # Statistics section
    st.markdown("---")
    st.subheader("📈 Global Health Data Statistics")
    st.write("### Data Statistics")
    st.write(data.describe())

    st.write("### Additional Insights")
    st.write(f"Total entries: {len(data)}")
    st.write(f"Columns: {', '.join(data.columns)}")

    st.markdown("---")
    st.write("### Data Correlation")

    if selected_option == "Normalized Data":
        # Normalized Data View Options
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

            # Drop the ID column if it exists

            if 'ID' in data.columns:

                data_without_id = data.drop(columns=['ID'])

            else:

                data_without_id = data

            correlation_matrix = data_without_id.corr()

            # Create a heatmap using Seaborn

            fig, ax = plt.subplots(figsize=(8, 6))  # Adjust size as needed

            sns.heatmap(

                correlation_matrix,

                annot=True,  # Display values in each cell

                cmap="coolwarm",  # Color map

                fmt=".2f",  # Format for displaying correlation values

                linewidths=0.5,  # Line width between cells

                ax=ax

            )

            st.pyplot(fig)

    else:
        st.warning("Correlation analysis is only available for Normalized Data.")

        toggle_state = st.toggle("Switch to Normalized Data View")
        if toggle_state:
            pass

else:
    st.error("The dataset is empty or could not be loaded.")
