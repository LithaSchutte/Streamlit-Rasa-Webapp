from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import SlotSet
import pandas as pd
import numpy as np
from fuzzywuzzy import process
import json

DATA_PATH = "../data/global_health.csv"

try:
    data = pd.read_csv(DATA_PATH)
    print(data.head())
except FileNotFoundError:
    print(f"File not found!")


class ActionGetAverage(Action):
    def name(self) -> str:
        return "action_calculate_column_stat"

    def run(self, dispatcher: CollectingDispatcher, tracker, domain) -> list:
        # Get the column name and action from the slots
        column_name = tracker.get_slot("column_name")
        value_action = tracker.get_slot("value_action")

        if not value_action:
            dispatcher.utter_message(
                text="I need to know whether you want the 'minimum', 'maximum', or 'mean' value. Can you specify?")
            return []

        # Validate column name
        columns = data.columns.tolist()
        match, score = process.extractOne(column_name, columns) if column_name else (None, 0)

        if score < 80 or not match:
            suggestions = ", ".join(columns[:5])  # Limit to the first 5 for brevity
            dispatcher.utter_message(
                text=f"I couldn't understand the column name."
                     f"Please provide a valid column name. Here are some suggestions: {suggestions}")
            return [SlotSet("column_name", None)]  # Clear invalid column_name slot

        # Validate value_action
        if value_action not in ["mean", "minimum", "maximum"]:
            dispatcher.utter_message(
                text=f"I can only calculate the 'minimum', 'maximum', or 'mean'. You requested '{value_action}'.")
            return [SlotSet("value_action", None)]  # Clear invalid value_action slot

        # Perform the calculation
        try:
            if value_action == "mean":
                result = data[match].mean()
            elif value_action == "minimum":
                result = data[match].min()
            elif value_action == "maximum":
                result = data[match].max()

            dispatcher.utter_message(text=f"The {value_action} for '{match}' is {result}")
            return [SlotSet("column_name", match), SlotSet("value_action", value_action)]
        except Exception as e:
            dispatcher.utter_message(
                text=f"An error occurred while processing your request: {str(e)}")
            return []

class ActionCompareCountries(Action):
    def name(self) -> str:
        return "action_country_comparison"

    def run(self, dispatcher: CollectingDispatcher, tracker, domain) -> list:
        try:
            # Retrieve the necessary slots
            column_name = tracker.get_slot("column_name")
            countries = tracker.get_slot("GPE")

            # Validate slots
            if not column_name:
                dispatcher.utter_message(text="I didn't get the column name. Please provide it.")
                return []
            if not countries or len(countries) < 2:
                dispatcher.utter_message(text="I need two countries to compare. Please provide them.")
                return []

            # Check for columns in the dataset
            columns = data.columns.tolist()
            if not columns:
                dispatcher.utter_message(text="The dataset appears to be empty or not loaded.")
                return []

            # Match the column name to the dataset columns
            match, score = process.extractOne(column_name, columns)
            if score < 70:  # Fuzzy matching threshold
                dispatcher.utter_message(text=f"I couldn't find a column similar to '{column_name}'. Please check the name.")
                return [SlotSet("column_name", None)]

            # Match both countries
            matched_countries = self.match_countries(countries)

            # Check if both countries were matched successfully
            if len(matched_countries) < 2:
                dispatcher.utter_message(text="One or both of the countries couldn't be matched. Please check the names.")
                return []

            # Retrieve and compare data for both countries
            value_1, year_1 = self.get_country_data(matched_countries[0], match)
            value_2, year_2 = self.get_country_data(matched_countries[1], match)

            if value_1 is None or value_2 is None:
                dispatcher.utter_message(text="Could not retrieve data for one or both countries. Please try again.")
                return []

            # Send the comparison message
            message = (
                f"In {year_1}, the {match.replace('_', ' ')} of {matched_countries[0]} was {value_1:.2f} "
                f"and the {match.replace('_', ' ')} of {matched_countries[1]} was {value_2:.2f}."
            )
            dispatcher.utter_message(text=message)

            # Return updated slots
            return [SlotSet("column_name", column_name), SlotSet("GPE", countries)]

        except Exception as e:
            dispatcher.utter_message(text=f"An error occurred: {str(e)}. Please try again.")
            return []

    def match_countries(self, countries: list) -> list:
        matched_countries = []
        for country in countries:
            columns = data["Country"].unique()
            match, score = process.extractOne(country, columns)
            if score > 70:
                matched_countries.append(match)
        return matched_countries

    def get_country_data(self, country: str, column: str):
        try:
            # Retrieve data for the country for the given column
            value = data[(data['Country'] == country) & (data['Year'] == 2021)][column].iloc[0]
            year = 2021
            if np.isnan(value):
                # Fallback to the previous year if 2021 data is missing
                value = data[(data['Country'] == country) & (data['Year'] == 2020)][column].iloc[0]
                year = 2020
            return value, year
        except (IndexError, KeyError, ValueError):
            return None, None


class ActionHealthByYear(Action):
    def name(self) -> str:
        return "action_health_year"

    def run(self, dispatcher: CollectingDispatcher, tracker, domain) -> list:
        try:
            # Retrieve slots
            column_name = tracker.get_slot("column_name")
            year_slot = tracker.get_slot("DATE")
            countries = tracker.get_slot("GPE")

            if not column_name or not year_slot or not countries:
                raise ValueError("Missing required slots: column_name, DATE, or GPE.")

            # Validate year
            try:
                year = int(year_slot)
            except ValueError:
                raise ValueError(f"Invalid year provided: {year_slot}")

            # Validate countries list
            if not isinstance(countries, list) or len(countries) == 0:
                raise ValueError("GPE slot must be a non-empty list of country names.")

            # Match column name
            columns = data.columns.tolist()
            match, score = process.extractOne(column_name, columns)

            data_countries = data["Country"].unique()
            match_country, score_country = process.extractOne(countries[0], data_countries)

            if not match:
                raise ValueError(f"No matching column found for: {column_name}")

            # Fetch value
            try:
                value = data[(data['Country'] == match_country) & (data['Year'] == year)][match].iloc[0]
            except (KeyError, IndexError):
                raise ValueError(f"Data not found for {match_country} in {year} for column {match}.")

            # Prepare and send message
            message = f"The value for {match} in {match_country} in {year} was {value:.2f}"
            dispatcher.utter_message(text=message)

            return [
                SlotSet("column_name", column_name),
                SlotSet("GPE", countries[0]),
                SlotSet("DATE", year)
            ]

        except ValueError as e:
            dispatcher.utter_message(text=str(e))
        except Exception as e:
            dispatcher.utter_message(text=f"An unexpected error occurred: {str(e)}")

        # Clear slots if something goes wrong
        return [
            SlotSet("column_name", None),
            SlotSet("GPE", None),
            SlotSet("DATE", None)
        ]


class ActionHealthDevelopment(Action):
    def name(self) -> str:
        return "action_health_development"

    def run(self, dispatcher: CollectingDispatcher, tracker, domain) -> list:

        column_name = tracker.get_slot("column_name")
        countries = tracker.get_slot("GPE")

        data_countries = data["Country"].unique()
        country, score = process.extractOne(countries[0], data_countries)
        columns = data.columns.tolist()

        match, score = process.extractOne(column_name, columns)

        filtered_data = data[data["Country"] == country]

        year_values = filtered_data[["Year", match]].dropna().values.tolist()

        message = f"Here are the values for {match} in {country} across all years:\n"
        for year, value in year_values:
            message += f"- {year}: {value:.2f}\n"

        dispatcher.utter_message(text=message)

        return [SlotSet("column_name", column_name), SlotSet("GPE", countries)]

class ActionGetCorrelation(Action):
    def name(self) -> str:
        return "action_get_correlation"

    def run(self, dispatcher: CollectingDispatcher, tracker, domain):

        # Load responses from the JSON file
        try:
            with open("responses.json", "r") as file:
                responses = json.load(file)
        except Exception as e:
            dispatcher.utter_message(text="Sorry, I couldn't load the responses.")
            return []

        # Extract the topic entity
        column_name = tracker.get_slot('column_name')
        columns = data.columns.tolist()
        matched, score = process.extractOne(column_name, columns)

        if matched and matched in responses:
            # Send the corresponding response
            dispatcher.utter_message(text=responses[matched])
        else:
            dispatcher.utter_message(text="I don't have information on that topic.")

        return [SlotSet("column_name", matched)]

