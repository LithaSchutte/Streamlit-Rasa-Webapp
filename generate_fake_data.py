import pandas as pd
import numpy as np
import random
import scipy.stats as stats

def count_decimal_places(num):  # helper function
    """Count the number of decimal places in a number."""
    num_str = str(num)
    if '.' in num_str:
        return len(num_str.split('.')[1].rstrip('0'))
    else:
        return 0

def generate_fake_row(df, country_code_pairs, columns_to_randomize):
    """Generate randomized, realistic fake row using information from real dataset."""
    fake_row = {}
    # Select random country and code
    country, code = random.choice(country_code_pairs)

    fake_row["Country"] = country
    fake_row["Country_Code"] = code
    fake_row["Year"] = random.choice(np.arange(2012, 2022))

    # Calculate life expectancy values first
    life_expectancy_female = stats.norm.rvs(
        loc=df["Life_Expectancy_Female"].mean(),
        scale=df["Life_Expectancy_Female"].std()
    )
    gap = random.uniform(4, 6)
    life_expectancy_male = life_expectancy_female - gap

    for column in columns_to_randomize:
        min_val, max_val = df[column].quantile([0.05, 0.95])  # Use 5th-95th percentile for bounds
        precision = count_decimal_places(df[column].iloc[0])  # Dynamically determine precision

        male_population = None
        female_population = None
        co2_exposure_value = None
        # Adjust Life Expectancy values
        if column == "Life_Expectancy":
            random_value = (life_expectancy_female + life_expectancy_male) / 2
        elif column == "Life_Expectancy_Female":
            random_value = life_expectancy_female
        elif column == "Life_Expectancy_Male":
            random_value = life_expectancy_male
        elif column == "Total_Population":
            # Generate Total Population but defer assigning Male/Female Population
            total_population = random.randint(int(min_val), int(max_val))
            random_value = total_population
            male_ratio = 102 / (100 + 102)
            male_population = round(total_population * male_ratio)
            female_population = total_population - male_population
        elif column == "Female_Population":
            random_value = female_population
        elif column == "Male_Population":
            random_value = male_population
        elif column == "CO2_Exposure_Percent":
            # Generate a single value for both columns
            co2_exposure_value = random.uniform(min_val, max_val)
            random_value = co2_exposure_value
        elif column == "Air_Pollution":
            random_value = co2_exposure_value
        else:
            if stats.shapiro(df[column])[1] > 0.05:
                random_value = stats.norm.rvs(
                    loc=df[column].mean(),
                    scale=df[column].std()
                )
                random_value = np.clip(random_value, min_val, max_val)
            else:
                random_value = random.uniform(min_val, max_val)

        fake_row[column] = round(random_value, precision)

    return fake_row

def generate_fake_data(original_data: pd.DataFrame, num_new_rows=1000) -> pd.DataFrame:
    """Generate n number of fake rows and return as a dataframe."""
    country_code_pairs = [[country, code] for code, country in zip(original_data["Country_Code"].unique(), original_data["Country"].unique())]
    columns_to_randomize = [col for col in original_data.columns if col not in ["Year", "Country", "Country_Code"]]

    new_rows = [generate_fake_row(original_data, country_code_pairs, columns_to_randomize) for _ in range(num_new_rows)]
    new_data = pd.DataFrame(new_rows)

    return new_data

def add_fake_data_to_real_data(real_data: pd.DataFrame, fake_data: pd.DataFrame) -> pd.DataFrame:
    """Join fake data with real/original data"""
    fake_dataset = real_data.copy()
    fake_dataset = pd.concat([fake_dataset, fake_data], ignore_index=True)
    return fake_dataset