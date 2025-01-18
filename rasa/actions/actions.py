from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import SlotSet
import pandas as pd
from fuzzywuzzy import process

DATA_PATH = "../data/global_health.csv"

try:
    data = pd.read_csv(DATA_PATH)
    print(data.head())
except FileNotFoundError:
    print(f"File not found!")

class ActionGetCountryData(Action):
    def name(self) -> str:
        return "action_country_specific"

    def run(self, dispatcher: CollectingDispatcher, tracker, domain) -> list:
        geo_country = tracker.get_slot("GPE")

        if not geo_country:
            dispatcher.utter_message(
                text="I didn't understand the country name. Can you please specify the country that you meant?")
            return []

        if len(geo_country) == 1:
            dispatcher.utter_message(text=f"The selected countries are: {geo_country[0]} and the len is {len(geo_country)}.")
        else:
            dispatcher.utter_message(text=f"The selected countries are: {geo_country[0]} and {geo_country[1]} and the len is {len(geo_country)}.")
        return [SlotSet("GPE", geo_country)]


class ActionGetAverage(Action):
    def name(self) -> str:
        return "action_calculate_column_stat"

    def run(self, dispatcher: CollectingDispatcher, tracker, domain) -> list:
        # Get the column name from the slot
        column_name = tracker.get_slot("column_name")
        value_action = tracker.get_slot("value_action")

        if not value_action:
            dispatcher.utter_message(
                text="I need to know whether you want the 'min', 'max', or 'mean' value. Can you specify?")
            return []

        columns = data.columns.tolist()
        match, score = process.extractOne(column_name, columns)

        if value_action == "mean":
            result = data[match].mean()
        elif value_action == "min":
            result = data[match].min()
        elif value_action == "max":
            result = data[match].max()
        else:
            dispatcher.utter_message(
                text=f"I can only calculate 'min', 'max', or 'mean'. You requested '{value_action}'.")
            return []

        if not column_name:
            dispatcher.utter_message(text="I couldn't understand the column name. Can you clarify?")
            return []

        # Simulate processing the column name
        dispatcher.utter_message(text=f"The {value_action} for {match} is {result}")
        return [SlotSet("column_name", column_name), SlotSet("value_action", value_action)]

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
        return "action_health_by_year"

    def run(self, dispatcher: CollectingDispatcher, tracker, domain) -> list:
        column_name = tracker.get_slot("column_name")
        year = tracker.get_slot("DATE")

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

