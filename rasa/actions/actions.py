from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import SlotSet
import pandas as pd
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
                text=f"I couldn't understand the column name '{column_name}'. "
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
        column_name = tracker.get_slot("column_name")

        columns = data.columns.tolist()
        match, score = process.extractOne(column_name, columns)

        countries = tracker.get_slot("GPE")

        value_1 = data[(data['Country'] == countries[0]) & (data['Year'] == 2021)][match].iloc[0]
        value_2 = data[(data['Country'] == countries[1]) & (data['Year'] == 2021)][match].iloc[0]

        message = f"In 2021 the {match} of {countries[0]} was {value_1:.2f} and the {match} of {countries[1]} was {value_2:.2f}"

        dispatcher.utter_message(text=message)

        return [SlotSet("column_name", column_name), SlotSet("GPE", countries)]

class ActionHealthByYear(Action):
    def name(self) -> str:
        return "action_health_year"

    def run(self, dispatcher: CollectingDispatcher, tracker, domain) -> list:
        column_name = tracker.get_slot("column_name")
        year = int(tracker.get_slot("DATE"))

        columns = data.columns.tolist()
        match, score = process.extractOne(column_name, columns) # match = column

        countries = tracker.get_slot("GPE")

        value = data[(data['Country'] == countries[0]) & (data['Year'] == year)][match].iloc[0]

        message = f"The value for {match} in {countries[0]} in {year} was {value:.2f}"

        dispatcher.utter_message(text=message)

        return [SlotSet("column_name", column_name), SlotSet("GPE", countries), "DATE", year]


class ActionHealthDevelopment(Action):
    def name(self) -> str:
        return "action_health_development"

    def run(self, dispatcher: CollectingDispatcher, tracker, domain) -> list:

        column_name = tracker.get_slot("column_name")
        countries = tracker.get_slot("GPE")

        country = countries[0]
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

        if column_name and column_name in responses:
            # Send the corresponding response
            dispatcher.utter_message(text=responses[column_name])
        else:
            dispatcher.utter_message(text="I don't have information on that topic.")

        return [SlotSet("column_name", column_name)]

