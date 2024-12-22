import pandas as pd
import numpy as np
import random
import streamlit as st

data_file = "global_health.csv"

@st.cache_data
def load_data(file_path):
    """Load the data from a CSV file."""
    data = pd.read_csv(file_path)
    return data


def count_decimal_places(num):
    """Count the number of decimal places in a number."""
    num_str = str(num)
    if '.' in num_str:
        return len(num_str.split('.')[1].rstrip('0'))
    else:
        return 0

def generate_fake_row(df, country_code_pairs, columns_to_randomize):
    fake_data = {}

    country, code = random.choice(country_code_pairs)

    fake_data["Country"] = country
    fake_data["Country_Code"] = code
    fake_data["Year"] = int(random.choice(np.arange(2012, 2022)))

    # respective male and female life expectancies need to be calculated before life expectancy, life expectancy is the average
    life_expectancy_female = random.uniform(df["Life_Expectancy_Female"].min(), df["Life_Expectancy_Female"].max())
    gap = random.uniform(4, 6)
    life_expectancy_male = life_expectancy_female - gap

    for column in columns_to_randomize:
        min_val = df[column].min()
        max_val = df[column].max()
        first_val = df[column].iloc[0]
        last_val = df[column].iloc[-1]

        # Determine the precision based on the first and last values to ensure consistency
        precision = max(
            len(str(first_val).split('.')[1]) if '.' in str(first_val) else 0,
            len(str(last_val).split('.')[1]) if '.' in str(last_val) else 0,
        )

        if column == "Life_Expectancy":
            random_value = (life_expectancy_female + life_expectancy_male) / 2  # calculate average
        elif column == "Life_Expectancy_Female":
            random_value = life_expectancy_female  # assign previously computed value
        elif column == "Life_Expectancy_Male":
            random_value = life_expectancy_male  # assign previously computed value
        else:
            random_value = random.uniform(min_val, max_val)

        fake_data[column] = float(round(random_value, precision))

    return fake_data


def generate_fake_data(original_data: pd.DataFrame, num_new_rows=1000) -> pd.DataFrame:
    df = load_data(original_data)

    country_code_pairs = [[country, code] for code, country in zip(df["Country_Code"].unique(), df["Country"].unique())]
    columns_to_randomize = [col for col in df.columns if col not in ["Year", "Country", "Country_Code"]]

    new_rows = [generate_fake_row(df, country_code_pairs, columns_to_randomize) for _ in range(num_new_rows)]
    new_data = pd.DataFrame(new_rows)

    return new_data


def add_fake_data_to_real_data(real_data: pd.DataFrame, fake_data: pd.DataFrame) -> pd.DataFrame:
    fake_dataset = real_data.copy()
    fake_dataset = pd.concat([fake_dataset, fake_data], ignore_index=True)
    return fake_dataset